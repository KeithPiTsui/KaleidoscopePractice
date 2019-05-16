#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

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
static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;
static std::map<std::string, Value*> NamedValues;

Value* LogErrorV(const char* Str){
  LogError(Str);
  return nullptr;
}

Value* NumberExprAST::codegen() {
  return ConstantFP::get(TheContext, APFloat(Val));
}

Value* VariableExprAST::codegen() {
  // Look this variable up in the fucntion
  Value* V = NamedValues[Name];
  if (!V) return LogErrorV("Unknown variable name");
  return V;
}

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
    return Builder.CreateUIToFP(L, Type::getDoubleTy(TheContext), "booltmp");
  default:
    return LogErrorV("invalid binary operator");
  }
}

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

Function* FunctionAST::codegen() {
  // First, check for an existing function from a previous 'extern' deaclaration.
  Function* TheFunction = TheModule->getFunction(Proto->getName());

  if (!TheFunction) TheFunction = Proto->codegen();
  if (!TheFunction) return nullptr;

  //Create a new basic block to start insertion into.
  BasicBlock* BB = BasicBlock::Create(TheContext, "entry", TheFunction);
  Builder.SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  for (auto& Arg : TheFunction->args())
    NamedValues[Arg.getName()] = &Arg;

  if (Value* RetVal = Body->codegen()) {
    Builder.CreateRet(RetVal);
    verifyFunction(*TheFunction);

    return TheFunction;
  }

  TheFunction->eraseFromParent();
  return nullptr;
}




//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//
static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto* FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition");
      FnIR->print(errs());
      fprintf(stderr, "\n");
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
    }
  } else {
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  if (auto FnAST = ParseTopLevelExpr()){
    if (auto* FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read a top-level expr\n");
      FnIR->print(errs());
      fprintf(stderr, "\n");
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
    case ';':
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


int main() {

  // Install standard binary operators.
  // 1 is lowest precedence
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.


  fprintf(stderr, "ready> ");
  getNextToken();

  TheModule = llvm::make_unique<Module>("my cool jit", TheContext);

  MainLoop();

  TheModule->print(errs(), nullptr);

  return 0;
}
