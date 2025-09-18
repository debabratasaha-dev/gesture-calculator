def calculate(expr: str):
    """
    Evaluate expression containing +, -, *, / with correct precedence.
    Handles unary +/- (e.g. "-5", "23--65"). Returns int when result is integral, else float.
    Raises ValueError on invalid input and ZeroDivisionError on division by zero.
    """
    expr = expr.replace(" ", "")
    if not expr:
        return 0.0
    while True:
        if expr[-1] in ("+", "-", "*", "/"):
            expr = expr[:-1]
        else:
            break

    # Tokenize: numbers (floats) and operators. Handle unary +/-
    tokens = []
    i = 0
    n = len(expr)
    ops = set("+-*/")

    while i < n:
        ch = expr[i]
        if ch.isdigit() or ch == ".":
            # parse number
            j = i
            while j < n and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            num_str = expr[i:j]
            try:
                num = float(num_str)
            except:
                raise ValueError(f"Invalid number: {num_str}")
            tokens.append(num)
            i = j
        elif ch in "+-":
            # Decide unary or binary:
            if i == 0 or (len(tokens) == 0) or (isinstance(tokens[-1], str) and tokens[-1] in ops):
                # unary: collect sequence of + and - to compute net sign
                sign = 1
                while i < n and expr[i] in "+-":
                    if expr[i] == "-":
                        sign *= -1
                    i += 1
                # next must be a number
                if i < n and (expr[i].isdigit() or expr[i] == "."):
                    j = i
                    while j < n and (expr[j].isdigit() or expr[j] == "."):
                        j += 1
                    num_str = expr[i:j]
                    try:
                        num = float(num_str) * sign
                    except:
                        raise ValueError(f"Invalid number: {num_str}")
                    tokens.append(num)
                    i = j
                else:
                    raise ValueError("Unary sign not followed by number")
            else:
                # binary + or -
                tokens.append(ch)
                i += 1
        elif ch in "*/":
            # binary * or /
            if len(tokens) == 0 or (isinstance(tokens[-1], str) and tokens[-1] in ops):
                raise ValueError(f"Unexpected operator {ch}")
            tokens.append(ch)
            i += 1
        else:
            raise ValueError(f"Illegal character: {ch}")

    if not tokens:
        raise ValueError("No tokens parsed")

    # First pass: handle * and /
    new_tokens = []
    i = 0
    while i < len(tokens):
        if isinstance(tokens[i], (int, float)):
            val = float(tokens[i])
            i += 1
            while i < len(tokens) and tokens[i] in ("*", "/"):
                op = tokens[i]; nxt = float(tokens[i+1])
                if op == "*":
                    val = val * nxt
                else:
                    if nxt == 0:
                        raise ZeroDivisionError("Division by zero")
                    val = val / nxt
                i += 2
            new_tokens.append(val)
        else:
            # shouldn't happen: operator in place of number
            raise ValueError("Syntax error in expression")
        if i < len(tokens):
            # next token should be + or -
            if tokens[i] in ("+", "-"):
                new_tokens.append(tokens[i])
                i += 1
            else:
                # if it's * or /, the previous loop would have consumed it
                continue

    # Second pass: evaluate + and - left-to-right
    result = float(new_tokens[0])
    i = 1
    while i < len(new_tokens):
        op = new_tokens[i]; nxt = float(new_tokens[i+1])
        if op == "+":
            result += nxt
        else:
            result -= nxt
        i += 2

    # return int when integral
    if abs(result - round(result)) < 1e-9:
        return int(round(result))
    return result