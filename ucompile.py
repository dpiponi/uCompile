#!/usr/bin/env python3
"""A super-minimal bc-like compiler that emits LLVM IR (.ll)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple


# ---------------------------
# Lexer
# ---------------------------

@dataclass
class Token:
    kind: str
    value: str
    pos: int


KEYWORDS = {"print", "if", "else", "while", "for", "define", "return"}
SINGLE_CHAR = set("+-*/%()=;{},[]\n<>!")


def lex(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(src)

    def add(kind: str, value: str, pos: int) -> None:
        tokens.append(Token(kind, value, pos))

    while i < n:
        c = src[i]
        # Comments: #..., //..., /* ... */
        if c == "#":
            i += 1
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i += 2
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            if i + 1 >= n:
                raise SyntaxError("Unterminated block comment")
            i += 2
            continue
        if c.isspace():
            i += 1
            continue
        if c in "+-*/" and i + 1 < n:
            nxt = src[i + 1]
            if nxt == "=":
                add(c + "=", c + "=", i)
                i += 2
                continue
            if c in "+-" and nxt == c:
                add(c + c, c + c, i)
                i += 2
                continue
        if c in SINGLE_CHAR:
            # Multi-char operators.
            if c in ("<", ">", "=", "!") and i + 1 < n and src[i + 1] == "=":
                add(c + "=", c + "=", i)
                i += 2
                continue
            add(c, c, i)
            i += 1
            continue
        if c == "\"":
            start = i
            i += 1
            out = []
            while i < n and src[i] != "\"":
                if src[i] == "\\":
                    i += 1
                    if i >= n:
                        break
                    esc = src[i]
                    if esc == "n":
                        out.append("\n")
                    elif esc == "t":
                        out.append("\t")
                    elif esc == "r":
                        out.append("\r")
                    elif esc == "\\":
                        out.append("\\")
                    elif esc == "\"":
                        out.append("\"")
                    else:
                        out.append(esc)
                    i += 1
                    continue
                out.append(src[i])
                i += 1
            if i >= n or src[i] != "\"":
                raise SyntaxError(f"Unterminated string at {start}")
            i += 1
            add("STRING", "".join(out), start)
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            start = i
            saw_dot = False
            while i < n and (src[i].isdigit() or src[i] == "."):
                if src[i] == ".":
                    if saw_dot:
                        break
                    saw_dot = True
                i += 1
            add("NUMBER", src[start:i], start)
            continue
        if c.isalpha() or c == "_":
            start = i
            i += 1
            while i < n and (src[i].isalnum() or src[i] == "_"):
                i += 1
            ident = src[start:i]
            if ident in KEYWORDS:
                add(ident, ident, start)
            else:
                add("IDENT", ident, start)
            continue
        raise SyntaxError(f"Unexpected character '{c}' at {i}")

    add("EOF", "", n)
    return tokens


# ---------------------------
# Parser
# ---------------------------

@dataclass
class Expr:
    pass


@dataclass
class Number(Expr):
    value: float


@dataclass
class Var(Expr):
    name: str


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class Stmt:
    pass


@dataclass
class Assign(Stmt):
    target: Expr
    expr: Expr


@dataclass
class AugAssign(Stmt):
    target: Expr
    op: str
    expr: Expr


@dataclass
class Print(Stmt):
    fmt: Optional[str]
    args: List[Expr]


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class If(Stmt):
    cond: Expr
    then_block: List[Stmt]
    else_block: Optional[List[Stmt]]


@dataclass
class While(Stmt):
    cond: Expr
    body: List[Stmt]


@dataclass
class For(Stmt):
    init: Optional[Stmt]
    cond: Optional[Expr]
    step: Optional[Stmt]
    body: List[Stmt]


@dataclass
class Return(Stmt):
    expr: Expr


@dataclass
class Call(Expr):
    name: str
    args: List[Expr]


@dataclass
class IncDec(Expr):
    target: Expr
    op: str
    pre: bool


@dataclass
class Index(Expr):
    name: str
    index: Expr


@dataclass
class FuncDef:
    name: str
    params: List[str]
    body: List[Stmt]


@dataclass
class Program:
    funcs: List[FuncDef]
    main: List[Stmt]


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.tokens[self.i]

    def next(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str) -> Token:
        tok = self.next()
        if tok.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {tok.kind} at {tok.pos}")
        return tok

    def parse(self) -> Program:
        funcs: List[FuncDef] = []
        main: List[Stmt] = []
        while self.peek().kind != "EOF":
            if self.peek().kind == ";":
                self.next()
                continue
            if self.peek().kind == "define":
                funcs.append(self.parse_funcdef())
            else:
                main.append(self.parse_stmt())
        return Program(funcs, main)

    def parse_stmt(self) -> Stmt:
        tok = self.peek()
        if tok.kind == "if":
            self.next()
            cond = self.parse_expr()
            then_block = self.parse_block_or_stmt()
            else_block = None
            if self.peek().kind == "else":
                self.next()
                else_block = self.parse_block_or_stmt()
            return If(cond, then_block, else_block)
        if tok.kind == "while":
            self.next()
            cond = self.parse_expr()
            body = self.parse_block_or_stmt()
            return While(cond, body)
        if tok.kind == "for":
            self.next()
            self.expect("(")
            init = None
            if self.peek().kind != ";":
                init = self.parse_for_part()
            self.expect(";")
            cond = None
            if self.peek().kind != ";":
                cond = self.parse_expr()
            self.expect(";")
            step = None
            if self.peek().kind != ")":
                step = self.parse_for_part()
            self.expect(")")
            body = self.parse_block_or_stmt()
            return For(init, cond, step, body)
        if tok.kind == "return":
            self.next()
            expr = self.parse_expr()
            self.expect(";")
            return Return(expr)
        if tok.kind == "print":
            self.next()
            if self.peek().kind == "STRING":
                fmt = self.next().value
                args: List[Expr] = []
                if self.peek().kind == ",":
                    self.next()
                    args.append(self.parse_expr())
                    while self.peek().kind == ",":
                        self.next()
                        args.append(self.parse_expr())
                self.expect(";")
                return Print(fmt, args)
            expr = self.parse_expr()
            args = [expr]
            while self.peek().kind == ",":
                self.next()
                args.append(self.parse_expr())
            self.expect(";")
            if len(args) == 1:
                return Print(None, args)
            return Print(None, args)
        if tok.kind == "IDENT":
            # Lookahead for assignment or index assignment
            if self.tokens[self.i + 1].kind in ("=", "+=", "-=", "*=", "/=", "["):
                saved = self.i
                name = self.next().value
                if self.peek().kind == "[":
                    self.next()
                    idx = self.parse_expr()
                    self.expect("]")
                    if self.peek().kind in ("=", "+=", "-=", "*=", "/="):
                        op = self.next().kind
                        expr = self.parse_expr()
                        self.expect(";")
                        if op == "=":
                            return Assign(Index(name, idx), expr)
                        return AugAssign(Index(name, idx), op, expr)
                    # Not an assignment; reset and parse as expression stmt.
                    self.i = saved
                if self.peek().kind in ("=", "+=", "-=", "*=", "/="):
                    op = self.next().kind
                    expr = self.parse_expr()
                    self.expect(";")
                    if op == "=":
                        return Assign(Var(name), expr)
                    return AugAssign(Var(name), op, expr)
        expr = self.parse_expr()
        self.expect(";")
        return ExprStmt(expr)

    def parse_for_part(self) -> Stmt:
        tok = self.peek()
        if tok.kind == "IDENT" and self.tokens[self.i + 1].kind in ("=", "+=", "-=", "*=", "/=", "["):
            name = self.next().value
            if self.peek().kind == "[":
                self.next()
                idx = self.parse_expr()
                self.expect("]")
                if self.peek().kind in ("=", "+=", "-=", "*=", "/="):
                    op = self.next().kind
                    expr = self.parse_expr()
                    if op == "=":
                        return Assign(Index(name, idx), expr)
                    return AugAssign(Index(name, idx), op, expr)
            if self.peek().kind in ("=", "+=", "-=", "*=", "/="):
                op = self.next().kind
                expr = self.parse_expr()
                if op == "=":
                    return Assign(Var(name), expr)
                return AugAssign(Var(name), op, expr)
        return ExprStmt(self.parse_expr())

    def parse_funcdef(self) -> FuncDef:
        self.expect("define")
        name = self.expect("IDENT").value
        self.expect("(")
        params: List[str] = []
        if self.peek().kind != ")":
            params.append(self.expect("IDENT").value)
            while self.peek().kind == ",":
                self.next()
                params.append(self.expect("IDENT").value)
        self.expect(")")
        body = self.parse_block()
        return FuncDef(name, params, body)

    def parse_block(self) -> List[Stmt]:
        self.expect("{")
        stmts: List[Stmt] = []
        while self.peek().kind != "}":
            stmts.append(self.parse_stmt())
        self.expect("}")
        return stmts

    def parse_block_or_stmt(self) -> List[Stmt]:
        if self.peek().kind == "{":
            return self.parse_block()
        return [self.parse_stmt()]

    def parse_expr(self) -> Expr:
        return self.parse_cmp()

    def parse_cmp(self) -> Expr:
        node = self.parse_add()
        while self.peek().kind in ("==", "!=", "<", "<=", ">", ">="):
            op = self.next().kind
            right = self.parse_add()
            node = BinOp(op, node, right)
        return node

    def parse_add(self) -> Expr:
        node = self.parse_mul()
        while self.peek().kind in ("+", "-"):
            op = self.next().kind
            right = self.parse_mul()
            node = BinOp(op, node, right)
        return node

    def parse_mul(self) -> Expr:
        node = self.parse_unary()
        while self.peek().kind in ("*", "/", "%"):
            op = self.next().kind
            right = self.parse_unary()
            node = BinOp(op, node, right)
        return node

    def parse_unary(self) -> Expr:
        if self.peek().kind in ("++", "--"):
            op = self.next().kind
            target = self.parse_unary()
            if not isinstance(target, (Var, Index)):
                raise SyntaxError("++/-- requires a variable or array element")
            return IncDec(target, op, True)
        if self.peek().kind in ("+", "-"):
            op = self.next().kind
            right = self.parse_unary()
            if op == "+":
                return right
            return BinOp("-", Number(0.0), right)
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.peek()
        if tok.kind == "NUMBER":
            self.next()
            return Number(float(tok.value))
        if tok.kind == "IDENT":
            name = self.next().value
            if self.peek().kind == "(":
                self.next()
                args: List[Expr] = []
                if self.peek().kind != ")":
                    args.append(self.parse_expr())
                    while self.peek().kind == ",":
                        self.next()
                        args.append(self.parse_expr())
                self.expect(")")
                expr: Expr = Call(name, args)
            else:
                if self.peek().kind == "[":
                    self.next()
                    idx = self.parse_expr()
                    self.expect("]")
                    expr = Index(name, idx)
                else:
                    expr = Var(name)
            if self.peek().kind in ("++", "--"):
                op = self.next().kind
                if not isinstance(expr, (Var, Index)):
                    raise SyntaxError("++/-- requires a variable or array element")
                return IncDec(expr, op, False)
            return expr
        if tok.kind == "(":
            self.next()
            expr = self.parse_expr()
            self.expect(")")
            return expr
        raise SyntaxError(f"Unexpected token {tok.kind} at {tok.pos}")


# ---------------------------
# LLVM IR Emitter
# ---------------------------

class IRBuilder:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.tmp = 0
        self.lbl = 0
        self.strings: Dict[str, str] = {}
        self.pending_globals: List[str] = []
        self.global_insert_idx: Optional[int] = None

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def fresh(self) -> str:
        self.tmp += 1
        return f"%t{self.tmp}"

    def fresh_label(self, base: str) -> str:
        self.lbl += 1
        return f"{base}{self.lbl}"

    def render(self) -> str:
        if self.pending_globals and self.global_insert_idx is not None:
            idx = self.global_insert_idx
            for line in self.pending_globals:
                self.lines.insert(idx, line)
                idx += 1
        return "\n".join(self.lines) + "\n"


class Compiler:
    def __init__(self) -> None:
        self.builder = IRBuilder()
        self.vars: Dict[str, str] = {}
        self.arrays: Dict[str, str] = {}
        self.entry_insert_idx: Optional[int] = None

    def compile(self, program: Program) -> str:
        b = self.builder

        b.emit('; ModuleID = "ucompile"')
        b.emit('%Array = type { double*, i64, i64 }')
        b.emit('declare i32 @printf(i8*, ...)')
        b.emit('declare i32 @rand()')
        b.emit('declare void @srand(i32)')
        b.emit('declare i8* @realloc(i8*, i64)')
        b.emit('declare void @llvm.memset.p0.i64(i8*, i8, i64, i1)')
        # Math builtins (libm / LLVM)
        b.emit('declare double @sin(double)')
        b.emit('declare double @cos(double)')
        b.emit('declare double @tan(double)')
        b.emit('declare double @asin(double)')
        b.emit('declare double @acos(double)')
        b.emit('declare double @atan(double)')
        b.emit('declare double @sinh(double)')
        b.emit('declare double @cosh(double)')
        b.emit('declare double @tanh(double)')
        b.emit('declare double @exp(double)')
        b.emit('declare double @log(double)')
        b.emit('declare double @log10(double)')
        b.emit('declare double @sqrt(double)')
        b.emit('declare double @pow(double, double)')
        b.emit('@.fmt = private constant [4 x i8] c"%g\\0A\\00"')
        b.emit('')
        b.global_insert_idx = len(b.lines)
        self.emit_array_helper()
        b.emit('')
        for func in program.funcs:
            self.emit_func(func)
            b.emit('')

        b.emit('define i32 @main() {')
        b.emit('entry:')
        self.vars = {}
        self.arrays = {}
        self.entry_insert_idx = len(b.lines)
        for stmt in program.main:
            self.emit_stmt(stmt)
        b.emit('  ret i32 0')
        b.emit('}')
        self.entry_insert_idx = None
        return b.render()

    def ensure_var(self, name: str) -> str:
        if name in self.arrays:
            raise ValueError(f"'{name}' is an array, not a scalar")
        if name in self.vars:
            return self.vars[name]
        slot = f"%{name}"
        self.insert_entry_lines(
            [
                f"  {slot} = alloca double",
                f"  store double 0.0, double* {slot}",
            ]
        )
        self.vars[name] = slot
        return slot

    def ensure_array(self, name: str) -> str:
        if name in self.vars:
            raise ValueError(f"'{name}' is a scalar, not an array")
        if name in self.arrays:
            return self.arrays[name]
        slot = f"%{name}"
        self.insert_entry_lines(
            [
                f"  {slot} = alloca %Array",
                f"  {slot}.data = getelementptr %Array, %Array* {slot}, i32 0, i32 0",
                f"  {slot}.size = getelementptr %Array, %Array* {slot}, i32 0, i32 1",
                f"  {slot}.cap = getelementptr %Array, %Array* {slot}, i32 0, i32 2",
                f"  store double* null, double** {slot}.data",
                f"  store i64 0, i64* {slot}.size",
                f"  store i64 0, i64* {slot}.cap",
            ]
        )
        self.arrays[name] = slot
        return slot

    def emit_stmt(self, stmt: Stmt) -> None:
        if isinstance(stmt, Assign):
            val = self.emit_expr(stmt.expr)
            if isinstance(stmt.target, Var):
                slot = self.ensure_var(stmt.target.name)
                self.builder.emit(f"  store double {val}, double* {slot}")
                return
            if isinstance(stmt.target, Index):
                ptr = self.emit_index_ptr(stmt.target)
                self.builder.emit(f"  store double {val}, double* {ptr}")
                return
            raise TypeError(f"Invalid assignment target: {type(stmt.target)}")
            return
        if isinstance(stmt, AugAssign):
            self.emit_augassign(stmt)
            return
        if isinstance(stmt, If):
            self.emit_if(stmt)
            return
        if isinstance(stmt, While):
            self.emit_while(stmt)
            return
        if isinstance(stmt, For):
            self.emit_for(stmt)
            return
        if isinstance(stmt, Return):
            val = self.emit_expr(stmt.expr)
            self.builder.emit(f"  ret double {val}")
            # Start a fresh label so following statements are in a new block.
            cont = self.builder.fresh_label("after_ret")
            self.builder.emit(f"{cont}:")
            return
        if isinstance(stmt, Print):
            if stmt.fmt is None:
                if len(stmt.args) == 1:
                    val = self.emit_expr(stmt.args[0])
                    fmt = self.builder.fresh()
                    self.builder.emit(
                        f"  {fmt} = getelementptr [4 x i8], [4 x i8]* @.fmt, i32 0, i32 0"
                    )
                    self.builder.emit(
                        f"  call i32 (i8*, ...) @printf(i8* {fmt}, double {val})"
                    )
                    return
                fmt_text = " ".join(["%g"] * len(stmt.args)) + "\\n"
                self.emit_printf(fmt_text, stmt.args)
                return
            self.emit_printf(stmt.fmt, stmt.args)
            return
        if isinstance(stmt, ExprStmt):
            self.emit_expr(stmt.expr)
            return
        raise TypeError(f"Unknown stmt type: {type(stmt)}")

    def emit_func(self, func: FuncDef) -> None:
        b = self.builder
        params_sig = ", ".join(f"double %{p}" for p in func.params)
        b.emit(f"define double @{func.name}({params_sig}) {{")
        b.emit("entry:")
        self.vars = {}
        self.arrays = {}
        self.entry_insert_idx = len(b.lines)
        for p in func.params:
            slot = f"%{p}"
            alloca = f"%{p}.addr"
            self.insert_entry_lines(
                [
                    f"  {alloca} = alloca double",
                    f"  store double {slot}, double* {alloca}",
                ]
            )
            self.vars[p] = alloca
        for s in func.body:
            self.emit_stmt(s)
        b.emit("  ret double 0.0")
        b.emit("}")
        self.entry_insert_idx = None

    def insert_entry_lines(self, lines: List[str]) -> None:
        if self.entry_insert_idx is None:
            raise RuntimeError("Entry insert position not set")
        for line in lines:
            self.builder.lines.insert(self.entry_insert_idx, line)
            self.entry_insert_idx += 1

    def emit_if(self, stmt: If) -> None:
        b = self.builder
        cond = self.emit_cond(stmt.cond)
        then_lbl = b.fresh_label("then")
        else_lbl = b.fresh_label("else") if stmt.else_block is not None else None
        end_lbl = b.fresh_label("endif")

        if else_lbl is None:
            b.emit(f"  br i1 {cond}, label %{then_lbl}, label %{end_lbl}")
        else:
            b.emit(f"  br i1 {cond}, label %{then_lbl}, label %{else_lbl}")

        b.emit(f"{then_lbl}:")
        for s in stmt.then_block:
            self.emit_stmt(s)
        b.emit(f"  br label %{end_lbl}")

        if else_lbl is not None:
            b.emit(f"{else_lbl}:")
            for s in stmt.else_block or []:
                self.emit_stmt(s)
            b.emit(f"  br label %{end_lbl}")

        b.emit(f"{end_lbl}:")

    def emit_while(self, stmt: While) -> None:
        b = self.builder
        cond_lbl = b.fresh_label("while_cond")
        body_lbl = b.fresh_label("while_body")
        end_lbl = b.fresh_label("while_end")

        b.emit(f"  br label %{cond_lbl}")
        b.emit(f"{cond_lbl}:")
        cond = self.emit_cond(stmt.cond)
        b.emit(f"  br i1 {cond}, label %{body_lbl}, label %{end_lbl}")

        b.emit(f"{body_lbl}:")
        for s in stmt.body:
            self.emit_stmt(s)
        b.emit(f"  br label %{cond_lbl}")

        b.emit(f"{end_lbl}:")

    def emit_for(self, stmt: For) -> None:
        b = self.builder
        if stmt.init is not None:
            self.emit_stmt(stmt.init)

        cond_lbl = b.fresh_label("for_cond")
        body_lbl = b.fresh_label("for_body")
        step_lbl = b.fresh_label("for_step")
        end_lbl = b.fresh_label("for_end")

        b.emit(f"  br label %{cond_lbl}")
        b.emit(f"{cond_lbl}:")
        if stmt.cond is None:
            b.emit(f"  br label %{body_lbl}")
        else:
            cond = self.emit_cond(stmt.cond)
            b.emit(f"  br i1 {cond}, label %{body_lbl}, label %{end_lbl}")

        b.emit(f"{body_lbl}:")
        for s in stmt.body:
            self.emit_stmt(s)
        b.emit(f"  br label %{step_lbl}")

        b.emit(f"{step_lbl}:")
        if stmt.step is not None:
            self.emit_stmt(stmt.step)
        b.emit(f"  br label %{cond_lbl}")

        b.emit(f"{end_lbl}:")

    def emit_cond(self, expr: Expr) -> str:
        b = self.builder
        if isinstance(expr, BinOp) and expr.op in ("==", "!=", "<", "<=", ">", ">="):
            left = self.emit_expr(expr.left)
            right = self.emit_expr(expr.right)
            tmp = b.fresh()
            pred = {
                "==": "oeq",
                "!=": "one",
                "<": "olt",
                "<=": "ole",
                ">": "ogt",
                ">=": "oge",
            }[expr.op]
            b.emit(f"  {tmp} = fcmp {pred} double {left}, {right}")
            return tmp
        val = self.emit_expr(expr)
        tmp = b.fresh()
        b.emit(f"  {tmp} = fcmp one double {val}, 0.0")
        return tmp

    def emit_expr(self, expr: Expr) -> str:
        b = self.builder
        if isinstance(expr, Number):
            # Emit literal directly.
            return self.format_double(expr.value)
        if isinstance(expr, Var):
            slot = self.ensure_var(expr.name)
            tmp = b.fresh()
            b.emit(f"  {tmp} = load double, double* {slot}")
            return tmp
        if isinstance(expr, Index):
            ptr = self.emit_index_ptr(expr)
            tmp = b.fresh()
            b.emit(f"  {tmp} = load double, double* {ptr}")
            return tmp
        if isinstance(expr, IncDec):
            return self.emit_incdec(expr)
        if isinstance(expr, Call):
            if expr.name == "rand" and len(expr.args) == 0:
                ri = b.fresh()
                b.emit(f"  {ri} = call i32 @rand()")
                rf = b.fresh()
                b.emit(f"  {rf} = uitofp i32 {ri} to double")
                tmp = b.fresh()
                b.emit(f"  {tmp} = fdiv double {rf}, 2147483647.0")
                return tmp
            if expr.name == "srand" and len(expr.args) == 1:
                seed = self.emit_expr(expr.args[0])
                si = b.fresh()
                b.emit(f"  {si} = fptosi double {seed} to i32")
                b.emit(f"  call void @srand(i32 {si})")
                return "0.0"
            args = [self.emit_expr(a) for a in expr.args]
            args_sig = ", ".join(f"double {a}" for a in args)
            tmp = b.fresh()
            b.emit(f"  {tmp} = call double @{expr.name}({args_sig})")
            return tmp
        if isinstance(expr, BinOp):
            left = self.emit_expr(expr.left)
            right = self.emit_expr(expr.right)
            tmp = b.fresh()
            if expr.op == "+":
                b.emit(f"  {tmp} = fadd double {left}, {right}")
            elif expr.op == "-":
                b.emit(f"  {tmp} = fsub double {left}, {right}")
            elif expr.op == "*":
                b.emit(f"  {tmp} = fmul double {left}, {right}")
            elif expr.op == "/":
                b.emit(f"  {tmp} = fdiv double {left}, {right}")
            elif expr.op == "%":
                # fmod via frem
                b.emit(f"  {tmp} = frem double {left}, {right}")
            elif expr.op in ("==", "!=", "<", "<=", ">", ">="):
                pred = {
                    "==": "oeq",
                    "!=": "one",
                    "<": "olt",
                    "<=": "ole",
                    ">": "ogt",
                    ">=": "oge",
                }[expr.op]
                b.emit(f"  {tmp} = fcmp {pred} double {left}, {right}")
                as_double = b.fresh()
                b.emit(f"  {as_double} = uitofp i1 {tmp} to double")
                return as_double
            else:
                raise ValueError(f"Unknown operator: {expr.op}")
            return tmp
        raise TypeError(f"Unknown expr type: {type(expr)}")

    def emit_incdec(self, expr: IncDec) -> str:
        b = self.builder
        ptr = self.emit_target_ptr(expr.target)
        old = b.fresh()
        b.emit(f"  {old} = load double, double* {ptr}")
        delta = "1.0"
        new = b.fresh()
        if expr.op == "++":
            b.emit(f"  {new} = fadd double {old}, {delta}")
        else:
            b.emit(f"  {new} = fsub double {old}, {delta}")
        b.emit(f"  store double {new}, double* {ptr}")
        return new if expr.pre else old

    def emit_augassign(self, stmt: AugAssign) -> None:
        b = self.builder
        ptr = self.emit_target_ptr(stmt.target)
        old = b.fresh()
        b.emit(f"  {old} = load double, double* {ptr}")
        rhs = self.emit_expr(stmt.expr)
        new = b.fresh()
        if stmt.op == "+=":
            b.emit(f"  {new} = fadd double {old}, {rhs}")
        elif stmt.op == "-=":
            b.emit(f"  {new} = fsub double {old}, {rhs}")
        elif stmt.op == "*=":
            b.emit(f"  {new} = fmul double {old}, {rhs}")
        elif stmt.op == "/=":
            b.emit(f"  {new} = fdiv double {old}, {rhs}")
        else:
            raise ValueError(f"Unknown aug-assign: {stmt.op}")
        b.emit(f"  store double {new}, double* {ptr}")

    def emit_target_ptr(self, target: Expr) -> str:
        if isinstance(target, Var):
            return self.ensure_var(target.name)
        if isinstance(target, Index):
            return self.emit_index_ptr(target)
        raise TypeError(f"Invalid target: {type(target)}")

    def emit_printf(self, fmt_text: str, args: List[Expr]) -> None:
        b = self.builder
        gname, length = self.get_string_global(fmt_text)
        fmt = b.fresh()
        b.emit(f"  {fmt} = getelementptr [{length} x i8], [{length} x i8]* {gname}, i32 0, i32 0")
        arg_vals = [self.emit_expr(a) for a in args]
        if arg_vals:
            arg_sig = ", ".join(f"double {a}" for a in arg_vals)
            b.emit(f"  call i32 (i8*, ...) @printf(i8* {fmt}, {arg_sig})")
        else:
            b.emit(f"  call i32 (i8*, ...) @printf(i8* {fmt})")

    def get_string_global(self, text: str) -> Tuple[str, int]:
        b = self.builder
        if text in b.strings:
            name = b.strings[text]
            length = self.string_len(text)
            return name, length
        name = f"@.str.{len(b.strings)}"
        data = self.escape_c_string(text) + "\\00"
        length = self.string_len(text)
        b.pending_globals.append(f"{name} = private constant [{length} x i8] c\"{data}\"")
        b.strings[text] = name
        return name, length

    @staticmethod
    def escape_c_string(text: str) -> str:
        out = []
        for ch in text:
            if ch == "\n":
                out.append("\\0A")
            elif ch == "\t":
                out.append("\\09")
            elif ch == "\r":
                out.append("\\0D")
            elif ch == "\"":
                out.append("\\22")
            elif ch == "\\":
                out.append("\\5C")
            else:
                c = ord(ch)
                if 32 <= c < 127:
                    out.append(ch)
                else:
                    out.append(f"\\{c:02X}")
        return "".join(out)

    @staticmethod
    def string_len(text: str) -> int:
        # +1 for NUL terminator
        return len(text.encode("utf-8")) + 1

    def emit_index_ptr(self, expr: Index) -> str:
        b = self.builder
        arr = self.ensure_array(expr.name)
        idx_val = self.emit_expr(expr.index)
        idx_i64 = b.fresh()
        b.emit(f"  {idx_i64} = fptosi double {idx_val} to i64")
        ptr = b.fresh()
        b.emit(f"  {ptr} = call double* @__uc_array_getptr(%Array* {arr}, i64 {idx_i64})")
        return ptr

    def emit_array_helper(self) -> None:
        b = self.builder
        b.emit("define double* @__uc_array_getptr(%Array* %arr, i64 %idx) {")
        b.emit("entry:")
        b.emit("  %data_ptr = getelementptr %Array, %Array* %arr, i32 0, i32 0")
        b.emit("  %size_ptr = getelementptr %Array, %Array* %arr, i32 0, i32 1")
        b.emit("  %cap_ptr = getelementptr %Array, %Array* %arr, i32 0, i32 2")
        b.emit("  %size = load i64, i64* %size_ptr")
        b.emit("  %cap = load i64, i64* %cap_ptr")
        b.emit("  %neg = icmp slt i64 %idx, 0")
        b.emit("  br i1 %neg, label %bad, label %okidx")
        b.emit("bad:")
        b.emit("  ret double* null")
        b.emit("okidx:")
        b.emit("  %need = icmp uge i64 %idx, %size")
        b.emit("  br i1 %need, label %grow, label %ok")
        b.emit("grow:")
        b.emit("  %new_size = add i64 %idx, 1")
        b.emit("  %capv = alloca i64")
        b.emit("  store i64 %cap, i64* %capv")
        b.emit("  %cap0 = load i64, i64* %capv")
        b.emit("  %cap_lt1 = icmp ult i64 %cap0, 1")
        b.emit("  br i1 %cap_lt1, label %cap_init, label %cap_grow")
        b.emit("cap_init:")
        b.emit("  store i64 1, i64* %capv")
        b.emit("  br label %cap_grow")
        b.emit("cap_grow:")
        b.emit("  %capcur = load i64, i64* %capv")
        b.emit("  %cap_ok = icmp uge i64 %capcur, %new_size")
        b.emit("  br i1 %cap_ok, label %cap_done, label %cap_inc")
        b.emit("cap_inc:")
        b.emit("  %cap2 = mul i64 %capcur, 2")
        b.emit("  store i64 %cap2, i64* %capv")
        b.emit("  br label %cap_grow")
        b.emit("cap_done:")
        b.emit("  %new_cap = load i64, i64* %capv")
        b.emit("  %data = load double*, double** %data_ptr")
        b.emit("  %need_realloc = icmp ne i64 %new_cap, %cap")
        b.emit("  br i1 %need_realloc, label %do_realloc, label %postalloc")
        b.emit("do_realloc:")
        b.emit("  %data_i8 = bitcast double* %data to i8*")
        b.emit("  %new_bytes = mul i64 %new_cap, 8")
        b.emit("  %new_i8 = call i8* @realloc(i8* %data_i8, i64 %new_bytes)")
        b.emit("  %new_data = bitcast i8* %new_i8 to double*")
        b.emit("  br label %postalloc")
        b.emit("postalloc:")
        b.emit("  %data_new = phi double* [ %new_data, %do_realloc ], [ %data, %cap_done ]")
        b.emit("  %delta = sub i64 %new_size, %size")
        b.emit("  %delta_bytes = mul i64 %delta, 8")
        b.emit("  %new_region = getelementptr double, double* %data_new, i64 %size")
        b.emit("  %new_region_i8 = bitcast double* %new_region to i8*")
        b.emit("  call void @llvm.memset.p0.i64(i8* %new_region_i8, i8 0, i64 %delta_bytes, i1 false)")
        b.emit("  store double* %data_new, double** %data_ptr")
        b.emit("  store i64 %new_size, i64* %size_ptr")
        b.emit("  store i64 %new_cap, i64* %cap_ptr")
        b.emit("  br label %ok")
        b.emit("ok:")
        b.emit("  %data2 = load double*, double** %data_ptr")
        b.emit("  %ptr = getelementptr double, double* %data2, i64 %idx")
        b.emit("  ret double* %ptr")
        b.emit("}")

    @staticmethod
    def format_double(val: float) -> str:
        # LLVM accepts decimal literals like 1.23 or 1.0. Keep it simple.
        if val == float("inf"):
            return "0x7FF0000000000000"
        if val == float("-inf"):
            return "0xFFF0000000000000"
        if val != val:  # NaN
            return "0x7FF8000000000000"
        s = repr(val)
        if "e" in s or "E" in s:
            # LLVM accepts scientific notation, keep as is.
            return s
        if "." not in s:
            s += ".0"
        return s


# ---------------------------
# CLI
# ---------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compile a tiny bc-like language to LLVM IR (.ll).")
    parser.add_argument("input", help="Input script file")
    parser.add_argument("-o", "--output", required=True, help="Output .ll file")
    args = parser.parse_args(argv)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            src = f.read()
        tokens = lex(src)
        parser_ = Parser(tokens)
        program = parser_.parse()
        compiler = Compiler()
        ir = compiler.compile(program)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(ir)
    except (SyntaxError, ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
