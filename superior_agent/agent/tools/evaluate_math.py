import ast
import operator

# Safe math mapping
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub, 
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg
}

def _eval_expr(node):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif hasattr(ast, 'Constant') and isinstance(node, ast.Constant): # python > 3.8
        return node.value
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return _OPERATORS[type(node.op)](_eval_expr(node.left), _eval_expr(node.right))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return _OPERATORS[type(node.op)](_eval_expr(node.operand))
    else:
        raise TypeError(node)

def evaluate_math(expression: str) -> str:
    """Description: Safely evaluates a mathematical expression using basic operators (+, -, *, /, **).
    Args: expression: The mathematical expression (e.g., '2 + 2', '10 * (5 - 3)').
    Returns: The numerical result.
    When to use: When asked to calculate, add, multiply, or solve math.
    """
    try:
        parsed = ast.parse(expression, mode='eval').body
        result = _eval_expr(parsed)
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression '{expression}': {exc}"
