from typing import Tuple, Callable


def make_function(name, parameters, body) -> Tuple[str, Callable]:
    declaration = "def {}({}):\n".format(name, parameters)
    declaration += "\n".join("    {}".format(statement) for statement in body)
    exec(declaration)
    func = locals()[name]
    return declaration, func


def compile_expression(expressions, name='preconditions') -> Tuple[str, Callable]:
    if len(expressions) == 0:
        body = 'return True'
    elif len(expressions) == 1:
        body = 'return ' + translate_expression(expressions[0])
    else:
        expr = " and ".join(translate_expression(e) for e in expressions)
        body = "return {}".format(expr)
    return make_function(name, 'state, constants', [body])


def compile_statements(statements, name='effects') -> Tuple[str, Callable]:
    if len(statements) == 0:
        body = ['pass']
    else:
        body = [translate_statement(stmt) for stmt in statements]
    return make_function(name, 'state, constants', body)


def check_numeric(token):
    try:
        float(token)
        return True
    except TypeError as e:
        raise TypeError(f'Got: {token}') from e
    except ValueError:
        return False


def state_var(tokens):
    return "state.state_vars[\"{}\"]".format(str(tokens))


def translate_expression(tokens):
    first_token = tokens[0] if isinstance(tokens, list) else tokens

    if check_numeric(first_token):
        return first_token
    elif first_token == '#t':
        return "constants.DELTA_T"
    elif first_token == 'or':
        expr = " or ".join(translate_expression(t) for t in tokens[1:])
        return "({})".format(expr)
    elif first_token == 'and':
        expr = " and ".join(translate_expression(t) for t in tokens[1:])
        return "({})".format(expr)
    elif first_token == 'not':
        return '(not {})'.format(translate_expression(tokens[1]))
    elif first_token in ['=', '>=', '<=', '>', '<']:
        first_token = '==' if first_token == '=' else first_token
        return "(round({}, constants.NUMBER_PRECISION) {} round({}, constants.NUMBER_PRECISION))". \
            format(translate_expression(tokens[1]), first_token, translate_expression(tokens[2]))
    elif first_token in ['+', '-', '*', '/']:
        return "round({} {} {}, constants.NUMBER_PRECISION)". \
            format(translate_expression(tokens[1]), first_token, translate_expression(tokens[2]))
    elif first_token == '^':
        return "round(pow({},{}), constants.NUMBER_PRECISION)". \
            format(translate_expression(tokens[1]), translate_expression(tokens[2]))
    else:
        # defer resolution to state
        return state_var(tokens)


def translate_statement(tokens):
    first_token = tokens[0] if isinstance(tokens, list) else tokens

    state_operators = {'assign': '=',
                       'increase': '+',
                       'decrease': '-',
                       'scale-up': '*',
                       'scale-down': '/'}

    if check_numeric(first_token):
        return first_token
    elif first_token in ['+', '-', '*', '/', '^', '=', '>=', '<=', '>', '<', '#t']:
        return translate_expression(tokens)
    elif first_token in state_operators.keys():
        if first_token == 'assign':
            return "{} {} round({}, constants.NUMBER_PRECISION)". \
                format(state_var(tokens[1]), state_operators[first_token], translate_expression(tokens[2]))
        return "{} = round({} {} {}, constants.NUMBER_PRECISION)". \
            format(state_var(tokens[1]), state_var(tokens[1]), state_operators[first_token], translate_expression(tokens[2]))
    elif first_token == 'not':
        return "{} = False".format(state_var(tokens[1]))
    else:
        return "{} = True".format(state_var(tokens))
