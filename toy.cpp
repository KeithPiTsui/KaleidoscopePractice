#include "KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

//===---------==//
// Lexer
//== ---------==//

// The lexer return tokens [0-255] if
// it is an unknown character, otherwise
// one of these for known things.

// The character set for this Kaleidoscope compile is ASCII.

enum Token {
            tok_eof = -1,

            // commands
            tok_def = -2,
            tok_extern = -3,

            // primary
            tok_identifier = -4,
            tok_number = -5
};

static std::string IdentifierStr; //Filled in if tok_identifier
static double NumVal; //Filled in if tok_number


// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';
  // Skip any whitespace.
  while(isspace(LastChar))
    // the C getchar() function to read characters one
    // at a time from standard input.
    // It eats them as it recognize them and stores
    // the last character read, but not processed, in
    // LastChar.
    LastChar = getchar();

  if (isalpha(LastChar)) {
    IdentifierStr = LastChar;
    while (isalnum(LastChar = getchar()))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return gettok();
  }

  // Check for end of file.
  // Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ASCII value.
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}



namespace {


  /// ExprAST - Base class for all expression nodes.
  class ExprAST {
  public:
    virtual ~ExprAST() = default;


    // The codegen() method says to emit IR for that AST node
    // along with all the things it depends on,
    // all they all return an LLVM Value object.
    // Value is the class used to represent a "Static Signle Assignment(SSA) register" or "SSA value"
    // in LLVM.
    // The most distinct aspect of SSA values is that their value is computed as the related instruction
    // executs, and it does not get new value until (and if) the instrcution re-executes.
    virtual Value *codegen() = 0;
  };


  /// NumberExprAST - Expression class for numeric literals like "1.0".
  class NumberExprAST : public ExprAST {
    double Val;

  public:
    NumberExprAST(double Val) : Val(Val) {}
    Value* codegen() override;
  };


  /// VariableExprAST - Expression class for referencing a variable, like "a".
  class VariableExprAST : public ExprAST {
    std::string Name;

  public:
    VariableExprAST(const std::string &Name) : Name(Name) {}
    Value* codegen() override;
  };


  /// BinaryExprAST - Expression class for a binary operator.
  class BinaryExprAST: public ExprAST {
    char Op;
    std::unique_ptr<ExprAST> LHS, RHS;

  public:
    BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                  std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
    Value* codegen() override;
  };


  /// CallExprAST - Expression class for function calls.
  class CallExprAST: public ExprAST {
    std::string Callee;
    std::vector<std::unique_ptr<ExprAST>> Args;

  public:
    CallExprAST(const std::string &Callee,
                std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}

    Value* codegen() override;
  };


  /// PrototypeAST - This class represents the "prototype" for a function,
  /// which captures its name, and its argument names (thus implicitly
  /// the number of arguments the function takes).
  class PrototypeAST{
    std::string Name;
    std::vector<std::string> Args;
  public:
    PrototypeAST(const std::string &Name, std::vector<std::string> Args)
      : Name(Name), Args(std::move(Args)) {}

    Function* codegen();
    const std::string &getName() const {return Name;}
  };


  /// FunctionAST - This class represents a function definition itself.
  class FunctionAST {
    std::unique_ptr<PrototypeAST> Proto;
    std::unique_ptr<ExprAST> Body;

  public:
    FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}
    Function* codegen();
  };

} // end anonymous namespace

//====-------====//
// Parser
//====-------====//


/// CurTok/getNextToken - Provide a simple token buffer.
/// CurTok is the current token the parser is looking at.
/// getNextToken reads another token from the lexer
/// and updates CurTok with its results.
/// This implements a simple token buffer around the lexer.
/// This allows us to look one token ahead at what the lexer
/// is returning.
/// Every function in our parser will assume that CurTok is
/// is the current token that needs to parsed.

static int CurTok;
static int getNextToken() {return CurTok = gettok(); }


// BinopPrecedence - This holds the precedence for each binary operator
// that is defined.
static std::map<char, int> BinopPrecedence;

// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  if(!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop
  int TokPrec = BinopPrecedence[CurTok];

  if (TokPrec <= 0)
    return -1;

  return TokPrec;
}


/// LogError* - These are little helper functions for error handling.
/// The LogError routins are simple helper routines that
// our parser will use to handle.

std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char* Str) {
  LogError(Str);
  return nullptr;
}


static std::unique_ptr<ExprAST> ParseExpression();


// for each production in our grammar, we'll define a function
// which parses that production.

// numberexpr ::= number
// it expects to be called whn the current token is a tok_number token.
// It takes the current number value, creates a NumberExprAST node,
// advances the lexer to the next token, and finally returns.

// The most important one is that this routine eats all of the tokens
// that correspond to the production and returns the lexer buffer with
// the next token (which is not part of the grammar production) ready to go.
// This is a fairly standard way to go for recursive descent parsers.

static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = llvm::make_unique<NumberExprAST>(NumVal);
  getNextToken();
  return std::move(Result);
}

// parenexpr ::= '(' expression ')'
// This function illustrates a number of interesting thins about the parser:
// 1) It shows how we use the LogError routines. When called, this fucntion
// expects that the current token is a '(' token, but after parsing the subexpression,
// it is possible that there is no ')' waiting. Because errors can occur,
// the parser needs a way to indicate that they happened: in our parser,
// we return null on an error.
// 2) Another interesting aspect of this function is that it uses recursion
// by calling ParseExpression. This is powerful because it allows us to
// handle recursive grammars, and keeps each production very simple.
// Note that parentheses do not cause construction of AST nodes themselves.
// While we could to it this way, the most important role of parentheses are to
// guide the parser and providing grouping. Once the parser constructs the AST,
// parentheses are not needed.

static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.

  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("expected ')'");

  getNextToken();//eat ).
  return V;
}


// identifier := identifier | identifier '(' expression* ')'
// One interesting aspect of this is that it uses lookahead
// to determine if the current identifier is a stand alone
// variable reference or if it is a function call expression.
// It handles this by checking to see if the token after the
// identifier is a '(' token, constructing either a VariableExprAST
// or CallExprAST node as appropriate
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier

  if (CurTok != '(' ) // Simple variable ref.
    return llvm::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while(true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError( "Expected ')' or ',' in argument list");

      getNextToken();
    }
  }

  // Eat the ')'
  getNextToken();

  return llvm::make_unique<CallExprAST>(IdName, std::move(Args));
}


// Primary ::= identifierexpr | numberexpr | parenexpr
// This uses look-ahead to determine which sort of expression is
// being inspected, and then parses it with a function call.
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch(CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  }
}


// binoprhs ::= ( '+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS) {

  // If this is binop, find its precedence.
  while(true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop
    int BinOp = CurTok;
    getNextToken(); // eat binop

    // Parse the primary expression after the binary operator.
    auto RHS = ParsePrimary();
    if (!RHS)
      return nullptr;

    // If Binop binds less tightly with RHS than the operator after RHS,
    // let the pending operator take RHS as its LHS
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS( TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS
    LHS = llvm::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
  } // loop around to the top of the while loop.
}


// expression := primary binoprhs
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParsePrimary();
  if (!LHS)
    return nullptr;
  return ParseBinOpRHS(0, std::move(LHS));
}

// prototype ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != tok_identifier)
    return LogErrorP("Expected function name in prototype");

  std::string FnName = IdentifierStr;

  getNextToken();

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  // Read the list of argument names.
  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if(CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.
  return llvm::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}


// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.

  auto Proto = ParsePrototype();

  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}


// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {

    // Make an anonymous proto.
    auto Proto = llvm::make_unique<PrototypeAST>("__anon_expr",
                                                 std::vector<std::string>());

    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}


// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken();
  return ParsePrototype();
}


//===--------------========//
// Code Generation
//===--------------========//

// The static variables will be used during code generation.
// TheContext is an opaque object that owns a lot of core LLVM data structures,
// such as the type and constant value tables.
static LLVMContext TheContext;

// The builder object is a helper object that makes it easy to generate LLVM instructions.
// Instances of the IRBuilder class template keep track of the current place to insert instructions
// and has methods to create new instructions.
static IRBuilder<> Builder(TheContext);

// TheModule is an LLVM construct that contains functions and global variables.
// In many ways, it is the top-level structure that the LLVM IR uses to contain code.
// It will own the memory for all of the IR that we generate, which is why the codegen() method
// returns a raw Value*, rather than a unique_ptr<Value>.
static std::unique_ptr<Module> TheModule;

// The NamedValues map keeps track of which values are defined in the current scope and what
// their LLVM representation is. (In other words, it is a symbol table for the code).
// In this form of Kaleidoscope, the only things that can be referenced are function parameters.
// As such, function parameters will be in this map when generating code for their function body.
static std::map<std::string, Value*> NamedValues;


static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;


Value* LogErrorV(const char* Str){
  LogError(Str);
  return nullptr;
}


Function* getFunction(std::string Name) {
  // Fisrt, see if the function has already been added to the current module.
  if (auto* F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return nil.
  return nullptr;
}

// In the LLVM IR, numeric constants are represented with the ConstatntFP class,
// with holds the numeric value in an APFloat internally (APFloat has the capability of holding floating
// point constants of Arbitrary Precision ).
// This code basically just creates and returns a ConstantFP.
// Note that in the LLVM IR that constants are all uniqued together and shared.
Value* NumberExprAST::codegen() {
  return ConstantFP::get(TheContext, APFloat(Val));
}


// In practice, the only values that can be in the NamedValues map are function arguments.
Value* VariableExprAST::codegen() {
  // Look this variable up in the fucntion
  Value* V = NamedValues[Name];
  if (!V) return LogErrorV("Unknown variable name");
  return V;
}


// The basic idea here is that we recursively emit code for the left-hand side
// of the expression, then the right-hand side,
// then we compute the result of the binary expression.
// In this code, we do a simple switch on the opcode to create the right LLVM instruction.
// IRBuilder knows where to insert the newly created instruction, all you have to do is
// specify what instruction to create (e.g. with CreateFAdd), which operands to use (L and R here)
// and optionally provide a name for the generated instruction.
// LLVM instructions are constrained by strict rules: for example, the Left and Right operators
// of an add instruction must have the same type, and the result type of the add must match the
// operand types.
// On the other hand, LLVM specifies that the fcmp instruction always return an 'i1' value
// (a one bit integer).
Value* BinaryExprAST::codegen() {
  Value* L = LHS->codegen();
  Value* R = RHS->codegen();
  if (!L || !R) return nullptr;

  switch(Op) {
  case '+':
    return Builder.CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder.CreateFSub(L, R, "subtmp");
  case '*':
    return Builder.CreateFMul(L, R, "multmp");
  case '<':
    L = Builder.CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return Builder.CreateUIToFP(L, Type::getDoubleTy(TheContext), "booltmp");
  default:
    return LogErrorV("invalid binary operator");
  }
}


// The code following initially does a function name lookup in the LLVM Module's symbol table.
// Recall that the LLVM Module is the container that holds the fucntions we are JIT'ing.
// By giving each function the same name as what the user specifies, we can use the LLVM symbol table
// to resolve function names for us.
// Once we have the function to call, we recursively codegen each argument that is to be passed in,
// and create an LLVM call instruction.
// Note that LLVM uses the native C calling convetions by default, allowing these calls to also
// call into standard library functions like "sin" or "cos" with no additional effort.
Value* CallExprAST::codegen() {
  // Loop up the name in the global module table.
  Function* CalleeF = TheModule->getFunction(Callee);
  if (!CalleeF) return LogErrorV("Unknown function referenceend");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value*> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back()) return nullptr;
  }

  return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}


// Note first that this function returns a "Function*" instead of a "Value*".
// Because a "prototype" really talks about the external interface for a function
// (not the value computed by an expression), it makes sense for it to return
// the LLVM Function it corresponds to when codegen'd.
// The call to FunctionType::get creates the FunctionType that should be used
// for a given Prototype. Since all function arguments in Kaleidoscope are of
// type double, the first line creates a vector of "N" LLVM double types. It then
// uses the FunctionType::get method to create a function type that takes "N" doubles
// as arguments, returns one double as a result, and that is not vararg (the false
// parameter indicates this).
// Note that Types in LLVM are uniqued just like Constants are, so you don't
// "new" a type, you "get" it.
// The final line above actually creates the IR Function corresponding to the Prototype.
// This indicates the type, linkage and name to use, as well as which module to insert into.
// "external linkage" means the function may be defined outside the current module and/or
// that it is callable by functions outside the module.
// The Name passed in is the name the user specified: since "TheModule" is specified, this
// name is registered in "TheModule"'s symbol table.
// Finally, we set the name of each of the function's arguments according to the names given
// in the Prototype.
// This step isn't strictly necessary, but keeping the names consistent makes the IR more
// readable, and allows subsequent code to refer directly to the arguments for their names,
// rather than having to look up in the Prototype AST.
// At this point we have a function prototype with no body.
// This is how LLVM IR represents function declarations.
// For extern statements in Kaleidoscope, this is as far as we need to go.
// For function definitions however, we need to codegen and attach a function body.
Function* PrototypeAST::codegen() {
  // make the function type: double(double, double) etc.
  std::vector<Type*> Doubles(Args.size(), Type::getDoubleTy(TheContext));
  FunctionType* FT =
    FunctionType::get(Type::getDoubleTy(TheContext), Doubles, false);

  Function* F =
    Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments
  unsigned Idx = 0;
  for (auto& Arg : F->args())
    Arg.setName(Args[Idx++]);

  return F;
}


// For function definitions, we start by searching TheModule's symbol table for an existing version
// of this function, in case one has already been created using an 'extern' statement.
// If Module::getFunction return null then no previous version exists, so we'll codegen
// one from the prototype.
// In either case, we want to assert that the function is empty (i.e. has no body yet) before we start.

Function* FunctionAST::codegen() {

  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below
  auto& P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function* TheFunction = getFunction(P.getName());
  if (!TheFunction) return nullptr;

  // Now we get to the point where the Builder is set up.
  // The first line creates a new basic block (named "entry"), which is inserted into TheFunction.
  // The second line then tells the builder that new instructions should be inserted into the
  // end of the new basic block.
  // Basic blocks in LLVM are an important part of functions that define the Control Flow Graph.
  // Since we don't have any control flow, our functions will only contain one block at this point.

  //Create a new basic block to start insertion into.
  BasicBlock* BB = BasicBlock::Create(TheContext, "entry", TheFunction);
  Builder.SetInsertPoint(BB);


  // Next we add the function arguments to the NamedValues map (after first clearing it out)
  // so that they're accessible to VariableExprAST nodes.
  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  for (auto& Arg : TheFunction->args())
    NamedValues[Arg.getName()] = &Arg;


  // Once the insertion point has been set up and the NamedValues map populated,
  // we call the codegen() method for the root expression of the function.
  // If no error happens, this emits code to compute the expression into the entry block
  // and returns the value that was computed.
  //Assuming no error, we then create an LLVM ret instruction, which completes the function.
  // Once the function is built, we call verifyFunction, which is provided by LLVM.
  // This function does a variety of consistency checks on the generated code,
  // to determine if our compiler is doing everything right.
  // Using this is import: it can catch a log of bugs.
  // Once the function is finished and validated, we return it.
  if (Value* RetVal = Body->codegen()) {
    // Finish off the function.
    Builder.CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    // Run the optimizer on the function.
    TheFPM->run(*TheFunction);

    return TheFunction;
  }
  // Error reading body, remove function.
  TheFunction->eraseFromParent();
  return nullptr;
}



//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void InitializeModuleAndPassManager() {
  // Open a new module.
  TheModule = llvm::make_unique<Module>("my cool jit", TheContext);
  TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

  // Create a new pass manager attached to it.
  TheFPM = llvm::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Do simple "peehole" optimizations and bit-twiddling optzns.
  TheFPM->add(createInstructionCombiningPass());

  // Reassociate expression.
  TheFPM->add(createReassociatePass());

  // Eliminate Common SubExression.
  TheFPM->add(createGVNPass());

  // Simplify the control flow graph (deleting unreachable blocks, etc).
  TheFPM->add(createCFGSimplificationPass());

  TheFPM->doInitialization();
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto* FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();
    }
  } else {
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto* FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read an extern\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-leve expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()){
    if (auto* FnIR = FnAST->codegen()){
      // JIT the module containing the anonymous expression,
      // keeping a handle so we can free it later.
      auto H = TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
      assert(ExprSymbol && "Function not found");

      // Get the symbol's address and cast it to the right type (takes no arguments,
      // returns a double) so we can call it as a native function.
      double (*FP)() = (double (*) ()) (intptr_t)cantFail(ExprSymbol.getAddress());
      fprintf(stderr, "Evaluated to %f\n", FP());

      // Delete the anonymous expression module from the JIT.
      TheJIT->removeModule(H);
    }
  } else {
    getNextToken();
  }
}

// top ::= definition | external | expression | ';'
static void MainLoop() {
  while(true) {
    fprintf(stderr, "ready> ");
    switch(CurTok) {
    case tok_eof:
      return;
    case ';':                   // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// putchard - putchar that takes a double and return 0.
extern "C" DLLEXPORT double putchard(double x){
  fputc((char) x, stderr);
  return 0;
}

// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double x) {
  fprintf(stderr, "%f\n", x);
  return 0;
}


//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main() {

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();



  // Install standard binary operators.
  // 1 is lowest precedence
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.


  // Prime the first toekn.
  fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = llvm::make_unique<KaleidoscopeJIT>();

  InitializeModuleAndPassManager();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
