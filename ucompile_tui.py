#!/usr/bin/env python3
"""Terminal UI for ucompile: edit left, IR on right (updates on valid code)."""

from __future__ import annotations

import curses
import os
import subprocess
import sys
import tempfile
import time
from typing import List, Tuple, Optional
import ctypes
import ctypes.util

ROOT = os.path.abspath(os.path.dirname(__file__))
COMPILER = os.path.join(ROOT, "ucompile.py")
try:
    from llvmlite import binding as llvm
except Exception:  # pragma: no cover - optional dependency
    llvm = None


def parse_error_line(src: str, err: str) -> Optional[int]:
    marker = " at "
    if marker not in err:
        return None
    try:
        pos = int(err.rsplit(marker, 1)[1])
    except ValueError:
        return None
    if pos < 0:
        return None
    line = src[:pos].count("\n")
    return line


def compile_source(src: str) -> Tuple[bool, str, Optional[int]]:
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input.bc")
        out_path = os.path.join(td, "out.ll")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(src)
        proc = subprocess.run(
            [sys.executable, COMPILER, in_path, "-o", out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "Compilation failed"
            return False, err, parse_error_line(src, err)
        with open(out_path, "r", encoding="utf-8") as f:
            return True, f.read(), None


def _init_llvm() -> Tuple[bool, str]:
    if llvm is None:
        return False, "llvmlite not installed (pip install -r requirements.txt)"
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    return True, ""


def _register_symbols() -> None:
    if llvm is None:
        return
    libc = ctypes.CDLL(None)
    libm_path = ctypes.util.find_library("m")
    libm = ctypes.CDLL(libm_path) if libm_path else None
    symbols = [
        "printf",
        "rand",
        "srand",
        "realloc",
        "free",
        "exit",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "sqrt",
        "pow",
    ]
    for name in symbols:
        addr = None
        if hasattr(libc, name):
            addr = ctypes.cast(getattr(libc, name), ctypes.c_void_p).value
        elif libm is not None and hasattr(libm, name):
            addr = ctypes.cast(getattr(libm, name), ctypes.c_void_p).value
        if addr:
            llvm.add_symbol(name, addr)


def _capture_stdout(func) -> Tuple[int, str, float]:
    # Redirect OS-level stdout to a temp file to capture printf output safely.
    with tempfile.TemporaryFile() as tf:
        old = os.dup(1)
        os.dup2(tf.fileno(), 1)
        start = time.perf_counter()
        rc = 1
        try:
            rc = func()
        finally:
            try:
                libc = ctypes.CDLL(None)
                if hasattr(libc, "fflush"):
                    libc.fflush(None)
            except Exception:
                pass
            elapsed = time.perf_counter() - start
            os.dup2(old, 1)
            os.close(old)
        tf.seek(0)
        out = tf.read().decode(errors="replace")
    return rc, out, elapsed


def run_ir(ir: str) -> Tuple[bool, str]:
    ok, msg = _init_llvm()
    if not ok:
        return False, msg
    _register_symbols()
    try:
        mod = llvm.parse_assembly(ir)
        mod.verify()
    except Exception as e:
        return False, f"IR verify failed: {e}"

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = llvm.create_mcjit_compiler(mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()

    addr = engine.get_function_address("main")
    if not addr:
        return False, "main not found"
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int)(addr)
    rc, out, elapsed = _capture_stdout(cfunc)
    if rc != 0:
        return False, f"Program exited with code {rc}"
    return True, out + format_duration(elapsed)


def format_duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"\n[time: {seconds * 1e9:.1f} ns]"
    if seconds < 1e-3:
        return f"\n[time: {seconds * 1e6:.1f} µs]"
    if seconds < 1.0:
        return f"\n[time: {seconds * 1e3:.1f} ms]"
    if seconds < 60.0:
        return f"\n[time: {seconds:.2f} s]"
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"\n[time: {mins}m {secs:.1f}s]"


def split_lines(text: str) -> List[str]:
    if not text:
        return [""]
    return text.split("\n")


def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def run(stdscr: "curses._CursesWindow", initial: str, path: str | None) -> None:
    curses.curs_set(1)
    stdscr.nodelay(False)
    stdscr.keypad(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)   # divider
    curses.init_pair(2, curses.COLOR_GREEN, -1)  # ok status
    curses.init_pair(3, curses.COLOR_RED, -1)    # error status
    curses.init_pair(4, curses.COLOR_YELLOW, -1) # mode/status info
    curses.init_pair(5, curses.COLOR_MAGENTA, -1) # keywords
    curses.init_pair(6, curses.COLOR_BLUE, -1)    # numbers
    curses.init_pair(7, curses.COLOR_GREEN, -1)   # comments
    curses.init_pair(8, curses.COLOR_CYAN, -1)    # operators

    buf = split_lines(initial)
    cur_y = 0
    cur_x = 0
    top = 0

    last_good_ir = ""
    last_good_out = ""
    last_error = ""
    error_line: Optional[int] = None
    stale = False
    show_ir = False
    last_save_msg = ""

    def current_source() -> str:
        return "\n".join(buf)

    def save_buffer() -> Tuple[bool, str]:
        if not path:
            return False, "No file path provided"
        with open(path, "w", encoding="utf-8") as f:
            f.write(current_source())
        return True, f"Saved {path}"

    # Initial compile
    ok, out, err_line = compile_source(current_source())
    if ok:
        last_good_ir = out
        run_ok, run_out = run_ir(out)
        if run_ok:
            last_good_out = run_out
            stale = False
            error_line = None
        else:
            last_error = run_out
            stale = True
            error_line = None
    else:
        last_error = out
        stale = True
        error_line = err_line

    def draw_line_with_syntax(y: int, x: int, text: str, max_w: int, in_block: bool) -> bool:
        if max_w <= 0:
            return in_block
        i = 0
        col = 0
        n = len(text)
        while i < n and col < max_w:
            c = text[i]
            nxt = text[i + 1] if i + 1 < n else ""

            if in_block:
                # Consume until end of block comment
                start = i
                while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                if i + 1 < n:
                    i += 2
                    in_block = False
                else:
                    i = n
                seg = text[start:i]
                stdscr.addnstr(y, x + col, seg, max_w - col, curses.color_pair(7))
                col += len(seg)
                continue

            # Line comment
            if c == "#":
                seg = text[i:]
                stdscr.addnstr(y, x + col, seg, max_w - col, curses.color_pair(7))
                break
            if c == "/" and nxt == "/":
                seg = text[i:]
                stdscr.addnstr(y, x + col, seg, max_w - col, curses.color_pair(7))
                break
            if c == "/" and nxt == "*":
                in_block = True
                stdscr.addnstr(y, x + col, "/*", max_w - col, curses.color_pair(7))
                i += 2
                col += 2
                continue

            # Numbers
            if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
                start = i
                saw_dot = False
                while i < n and (text[i].isdigit() or text[i] == "."):
                    if text[i] == ".":
                        if saw_dot:
                            break
                        saw_dot = True
                    i += 1
                seg = text[start:i]
                stdscr.addnstr(y, x + col, seg, max_w - col, curses.color_pair(6))
                col += len(seg)
                continue

            # Identifiers / keywords
            if c.isalpha() or c == "_":
                start = i
                i += 1
                while i < n and (text[i].isalnum() or text[i] == "_"):
                    i += 1
                seg = text[start:i]
                if seg in ("print", "if", "else", "while", "for", "define", "return"):
                    attr = curses.color_pair(5)
                else:
                    attr = 0
                stdscr.addnstr(y, x + col, seg, max_w - col, attr)
                col += len(seg)
                continue

            # Operators / punctuation
            if c in "+-*/%()=;{},[]<>!":
                stdscr.addnstr(y, x + col, c, max_w - col, curses.color_pair(8))
                i += 1
                col += 1
                continue

            # Default
            stdscr.addnstr(y, x + col, c, max_w - col)
            i += 1
            col += 1
        return in_block

    def safe_addnstr(y: int, x: int, s: str, attr: int = 0) -> None:
        if x < 0 or y < 0:
            return
        rows, cols = stdscr.getmaxyx()
        if y >= rows or x >= cols:
            return
        max_w = cols - x - 1
        if max_w <= 0:
            return
        try:
            stdscr.addnstr(y, x, s, max_w, attr)
        except curses.error:
            pass

    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        left_w = max(20, cols // 2)
        right_w = cols - left_w - 1
        edit_h = rows - 1

        # Draw divider
        for r in range(edit_h):
            if left_w < cols:
                stdscr.addch(r, left_w, ord("|"), curses.color_pair(1))

        # Render left editor with syntax highlighting
        in_block = False
        for i in range(edit_h):
            line_idx = top + i
            if line_idx >= len(buf):
                break
            line = buf[line_idx]
            in_block = draw_line_with_syntax(i, 0, line, left_w - 1, in_block)
            if error_line is not None and line_idx == error_line:
                stdscr.addnstr(i, max(0, left_w - 2), "!", 1, curses.color_pair(3))

        # Render right output (IR or program output)
        right_text = last_good_ir if show_ir else last_good_out
        ir_lines = split_lines(right_text)
        for i in range(edit_h):
            if i >= len(ir_lines):
                break
            line = ir_lines[i]
            if right_w > 0:
                stdscr.addnstr(i, left_w + 1, line, right_w)

        # Status bar
        status = "CTRL+G: quit | CTRL+O: toggle IR/output | CTRL+X: save"
        mode = "MODE: IR" if show_ir else "MODE: OUTPUT"
        safe_addnstr(rows - 1, 0, status, curses.color_pair(4))
        mode_x = len(status) + 3
        safe_addnstr(rows - 1, mode_x, mode, curses.color_pair(4))
        if stale:
            err = "ERROR (stale): " + last_error
            if error_line is not None:
                err += f" (line {error_line + 1})"
            err_x = mode_x + len(mode) + 3
            safe_addnstr(rows - 1, err_x, err, curses.color_pair(3))
        else:
            ok = "OK"
            ok_x = mode_x + len(mode) + 3
            safe_addnstr(rows - 1, ok_x, ok, curses.color_pair(2))
        if last_save_msg:
            msg_x = cols - len(last_save_msg) - 1
            safe_addnstr(rows - 1, msg_x, last_save_msg, curses.color_pair(2))

        # Cursor position
        screen_y = cur_y - top
        if 0 <= screen_y < edit_h:
            stdscr.move(screen_y, clamp(cur_x, 0, left_w - 2))

        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (ord("\x07"), 7):  # Ctrl+G
            break
        elif ch in (ord("\x0f"),):  # Ctrl+O
            show_ir = not show_ir
        elif ch in (ord("\x18"), 24):  # Ctrl+X
            ok, msg = save_buffer()
            last_save_msg = msg
        elif ch in (ord("\x01"), 1):  # Ctrl+A
            cur_x = 0
        elif ch in (ord("\x05"), 5):  # Ctrl+E
            cur_x = len(buf[cur_y])
        elif ch in (curses.KEY_LEFT,):
            if cur_x > 0:
                cur_x -= 1
            elif cur_y > 0:
                cur_y -= 1
                cur_x = len(buf[cur_y])
        elif ch in (curses.KEY_RIGHT,):
            if cur_x < len(buf[cur_y]):
                cur_x += 1
            elif cur_y + 1 < len(buf):
                cur_y += 1
                cur_x = 0
        elif ch in (curses.KEY_UP,):
            if cur_y > 0:
                cur_y -= 1
                cur_x = min(cur_x, len(buf[cur_y]))
        elif ch in (curses.KEY_DOWN,):
            if cur_y + 1 < len(buf):
                cur_y += 1
                cur_x = min(cur_x, len(buf[cur_y]))
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if cur_x > 0:
                line = buf[cur_y]
                buf[cur_y] = line[: cur_x - 1] + line[cur_x:]
                cur_x -= 1
            elif cur_y > 0:
                prev_len = len(buf[cur_y - 1])
                buf[cur_y - 1] += buf[cur_y]
                del buf[cur_y]
                cur_y -= 1
                cur_x = prev_len
        elif ch in (curses.KEY_DC,):
            line = buf[cur_y]
            if cur_x < len(line):
                buf[cur_y] = line[:cur_x] + line[cur_x + 1 :]
            elif cur_y + 1 < len(buf):
                buf[cur_y] += buf[cur_y + 1]
                del buf[cur_y + 1]
        elif ch in (10, 13):  # Enter
            line = buf[cur_y]
            left = line[:cur_x]
            right = line[cur_x:]
            buf[cur_y] = left
            buf.insert(cur_y + 1, right)
            cur_y += 1
            cur_x = 0
        elif ch == 9:  # Tab
            line = buf[cur_y]
            buf[cur_y] = line[:cur_x] + "  " + line[cur_x:]
            cur_x += 2
        elif 32 <= ch <= 126:
            line = buf[cur_y]
            buf[cur_y] = line[:cur_x] + chr(ch) + line[cur_x:]
            cur_x += 1

        # Keep cursor in view
        if cur_y < top:
            top = cur_y
        elif cur_y >= top + edit_h:
            top = cur_y - edit_h + 1

        # Recompile on any keypress that could change buffer
        if ch not in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT):
            ok, out, err_line = compile_source(current_source())
            if ok:
                last_good_ir = out
                run_ok, run_out = run_ir(out)
                if run_ok:
                    last_good_out = run_out
                    stale = False
                    last_error = ""
                    error_line = None
                else:
                    stale = True
                    last_error = run_out
                    error_line = None
            else:
                stale = True
                last_error = out
                error_line = err_line


def main() -> int:
    initial = ""
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            initial = f.read()
    curses.wrapper(run, initial, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
