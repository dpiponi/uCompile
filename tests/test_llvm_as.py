import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPILER = os.path.join(ROOT, "ucompile.py")
LLVM_AS = shutil.which("llvm-as")


@unittest.skipUnless(LLVM_AS, "llvm-as not found in PATH")
class TestLlvmAssemble(unittest.TestCase):
    def compile_and_assemble(self, src: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "input.bc")
            ll_path = os.path.join(td, "out.ll")
            bc_path = os.path.join(td, "out.bc")

            with open(src_path, "w", encoding="utf-8") as f:
                f.write(src)

            subprocess.run(
                [COMPILER, src_path, "-o", ll_path],
                check=True,
                cwd=ROOT,
            )
            subprocess.run([LLVM_AS, ll_path, "-o", bc_path], check=True)
            self.assertTrue(os.path.exists(bc_path))

    def test_basic_arith(self):
        self.compile_and_assemble(
            "x = 2 + 3 * 4;\n"
            "print x;\n"
            "print (x - 1) / 2;\n"
        )

    def test_vars_and_mod(self):
        self.compile_and_assemble(
            "a = 7;\n"
            "b = 3;\n"
            "print a % b;\n"
            "print a / b;\n"
        )

    def test_if_else(self):
        self.compile_and_assemble(
            "x = 4;\n"
            "if x > 3 {\n"
            "  print x;\n"
            "} else {\n"
            "  print 0;\n"
            "}\n"
        )

    def test_if_without_else(self):
        self.compile_and_assemble(
            "x = 1;\n"
            "if x { print 1; }\n"
            "print 2;\n"
        )

    def test_comparisons(self):
        self.compile_and_assemble(
            "x = 2;\n"
            "y = 3;\n"
            "print x < y;\n"
            "print x <= y;\n"
            "print x > y;\n"
            "print x >= y;\n"
            "print x == y;\n"
            "print x != y;\n"
        )

    def test_while(self):
        self.compile_and_assemble(
            "x = 3;\n"
            "while x > 0 {\n"
            "  print x;\n"
            "  x = x - 1;\n"
            "}\n"
        )

    def test_while_with_expr_cond(self):
        self.compile_and_assemble(
            "x = 3;\n"
            "while x {\n"
            "  x = x - 1;\n"
            "}\n"
            "print x;\n"
        )

    def test_for_loop(self):
        self.compile_and_assemble(
            "sum = 0;\n"
            "for (i = 0; i < 4; i = i + 1) {\n"
            "  sum = sum + i;\n"
            "}\n"
            "print sum;\n"
        )

    def test_for_loop_no_cond(self):
        self.compile_and_assemble(
            "i = 0;\n"
            "for (; i < 2; i = i + 1) { print i; }\n"
        )

    def test_functions(self):
        self.compile_and_assemble(
            "define add(a, b) {\n"
            "  return a + b;\n"
            "}\n"
            "x = add(2, 3);\n"
            "print x;\n"
        )

    def test_functions_nested_calls(self):
        self.compile_and_assemble(
            "define add(a, b) { return a + b; }\n"
            "define mul(a, b) { return a * b; }\n"
            "print mul(add(1, 2), 4);\n"
        )

    def test_recursion_fib(self):
        self.compile_and_assemble(
            "define fib(n) {\n"
            "  if n <= 1 {\n"
            "    return n;\n"
            "  } else {\n"
            "    return fib(n - 1) + fib(n - 2);\n"
            "  }\n"
            "}\n"
            "print fib(6);\n"
        )

    def test_function_default_return(self):
        self.compile_and_assemble(
            "define f(a) { a = a + 1; }\n"
            "print f(1);\n"
        )

    def test_arrays(self):
        self.compile_and_assemble(
            "a[0] = 1;\n"
            "a[1] = 2;\n"
            "print a[0] + a[1];\n"
        )

    def test_array_default_zero(self):
        self.compile_and_assemble(
            "print a[0];\n"
            "a[0] = a[0] + 1;\n"
            "print a[0];\n"
        )

    def test_arrays_with_expr_index(self):
        self.compile_and_assemble(
            "i = 1;\n"
            "a[i] = 10;\n"
            "a[i+1] = 20;\n"
            "print a[0] + a[1] + a[2];\n"
        )

    def test_unary_and_precedence(self):
        self.compile_and_assemble(
            "x = -1 + 2 * 3;\n"
            "y = -(1 + 2) * 3;\n"
            "print x;\n"
            "print y;\n"
        )

    def test_transcendentals(self):
        self.compile_and_assemble(
            "x = sin(0);\n"
            "y = cos(0);\n"
            "z = sqrt(4);\n"
            "w = pow(2, 3);\n"
            "print x + y + z + w;\n"
            "print log(1);\n"
            "print exp(1);\n"
        )

    def test_print_format(self):
        self.compile_and_assemble(
            "x = 1.5;\n"
            "y = 2.5;\n"
            "print \"x=%g y=%g\\n\", x, y;\n"
            "print \"hello\";\n"
            "print x, y, x + y;\n"
        )

    def test_rand_srand(self):
        self.compile_and_assemble(
            "srand(1);\n"
            "print rand();\n"
            "print rand();\n"
        )

    def test_augassign_incdec(self):
        self.compile_and_assemble(
            "x = 1;\n"
            "x += 2;\n"
            "x *= 3;\n"
            "x -= 4;\n"
            "x /= 2;\n"
            "print x;\n"
            "i = 0;\n"
            "i++;\n"
            "++i;\n"
            "print i;\n"
            "a[0] = 1;\n"
            "a[0]++;\n"
            "print a[0];\n"
        )


if __name__ == "__main__":
    unittest.main()
