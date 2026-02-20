# uCompile

A tiny, bc-like compiler that emits LLVM IR (`.ll`).

## Features (minimal)
- Numbers (floating point)
- Variables (identifiers)
- Arithmetic: `+ - * / %`
- Parentheses
- Assignment: `x = 1 + 2;`
- Print: `print x;`
- Conditionals: `if <expr> { ... } else { ... }`
- While loops: `while <expr> { ... }`
- For loops: `for (i = 0; i < 10; i = i + 1) { ... }`
- Functions: `define name(a, b) { return a + b; }`
- Arrays (1-D, auto-expanding, default 0): `a[0] = 1; print a[0];`
- Comments: `# ...`, `// ...`, `/* ... */`
- Transcendentals: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `log10`, `sqrt`, `pow`
- Print formatting: `print "x=%g y=%g\n", x, y;` and `print x, y, x+y;`
- Random: `rand()` in `[0,1)` and `srand(seed)`

## Usage
```bash
./ucompile.py input.bc -o out.ll
```

## Example
`example.bc`
```bc
x = 2 + 3 * 4;
print x;
print (x - 1) / 2;
if x > 10 {
  print x;
} else {
  print 0;
}
while x > 0 {
  print x;
  x = x - 1;
}
define add(a, b) {
  return a + b;
}
print add(2, 3);
a[0] = 1;
a[1] = 2;
print a[0] + a[1];
```

Compile to LLVM IR:
```bash
./ucompile.py example.bc -o example.ll
```

You can assemble and run with LLVM tools (if installed):
```bash
llc example.ll -o example.s
clang example.s -o example
./example
```

## Notes
- This is intentionally small and designed to be extended later with conditionals and functions.
- The backend emits a single `main` function and uses `printf` for `print`.

## Tests
These tests require `llvm-as` in your PATH.

Run:
```bash
python -m unittest -v
```

## Example: 1D Poisson (Jacobi)
See `examples/poisson1d.bc` for a small 1D Poisson solve using Jacobi iteration.

## UI (Terminal)
A minimal split-view TUI that edits on the left and shows output or LLVM IR on the right:

```bash
./ucompile_tui.py [path/to/file.bc]
```

- The right panel updates only on successful compilation (and execution for output mode).
- If the code is invalid, the right panel stays stale and the status bar shows the error.
- Default view shows program output; toggle with `Ctrl+O` to show IR.
- Save with `Ctrl+X`.
- Quit with `Ctrl+G`.

### Requirements for Output Mode
The TUI uses `llvmlite` to JIT and execute the compiled IR in-process for the output panel.

Install:
```bash
pip install -r requirements.txt
```

## Example: MLP for y = x^2
See `examples/mlp_x2.bc` for a simple 1-hidden-layer MLP trained with SGD to fit `y = x^2` on `[-1, 1]`.
