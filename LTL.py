
from ply.lex import lex
from ply.yacc import yacc

import numpy as np

def check_LTL(formula, trajectory, predicates):
  """
  a recursive function to evaluate an LTL formula over a tragectory. (for now only supports /\, \/, ~, >, ->, U, <> and []).
  Note on formula structure: Formula should be a nested list of operators/operands.
  each member is either a list of (evaluated) booleans, or an unevaluated tuple of shape: ('operator','operand', [other operands]) like ('^', 'sth' ) for eventually sth.
  or (' /\', 'sth1', 'sth2') for sth1  /\ sth2. also for predicates, the operator is 'None'.
  Example: '<>[]a' (eventually always a) -> the formula should look like this: ('<>', ('[]', (None, 'a') ) )

  inputs:
  formula: is the LTL formula that we want to evaluate. it can include: '<>' for eventually and '[]' for always.
  trajectory: is the history of the visited states by the agent. we evaluate the formula over this (example: [1,4,7,3,7,...])
  predicates: a dict of predicates used in the formula, with their corresponding True conditions. (example: {'a':[1,3,4,...]})

  outputs:
  a sequence of evaluations of the input formula over every time step of the trajectory. (example: [False, True, False,...])
  """

  # check if current formula needs evaluation
  if type(formula)==list: # already evaluated
    return formula
  
  # check if current formula is a predicate or not
  if formula[0]==None: # a predicate
    # need to evaluate predicate over the trajectory, and return it
    evaluation = []
    for i in trajectory:
      eval = i in predicates[formula[1]]
      evaluation.append(eval)
    return evaluation
  
  # the formula is an unevaluated operator
  # now evaluate the operator over the operand(s)
  evaluation = []
  if formula[0]=='<>':
    # Evaluate the operand.
    evaluated_operand = check_LTL(formula[1], trajectory, predicates)
    evaluation = [True in evaluated_operand[i:] for i in range(0,len(evaluated_operand))]

  elif formula[0]=='[]':
    # Evaluate the operand.
    evaluated_operand = check_LTL(formula[1], trajectory, predicates)
    evaluation = [all(evaluated_operand[i:]) for i in range(0,len(evaluated_operand))]

  elif formula[0]=='/\\':
    # Evaluate the operands.
    evaluated_operand1 = check_LTL(formula[1], trajectory, predicates)
    evaluated_operand2 = check_LTL(formula[2], trajectory, predicates)
    min_len_operand = min(len(evaluated_operand1), len(evaluated_operand2))
    evaluation = [all([evaluated_operand1[i],evaluated_operand2[i]]) for i in range(0, min_len_operand)]

  elif formula[0]=='\\/':
    # Evaluate the operands.
    evaluated_operand1 = check_LTL(formula[1], trajectory, predicates)
    evaluated_operand2 = check_LTL(formula[2], trajectory, predicates)
    min_len_operand = min(len(evaluated_operand1), len(evaluated_operand2))
    evaluation = [any([evaluated_operand1[i],evaluated_operand2[i]]) for i in range(0, min_len_operand)]

  elif formula[0]=='~':
    # Evaluate the operand.
    evaluated_operand = check_LTL(formula[1], trajectory, predicates)
    evaluation = [not i for i in evaluated_operand]

  elif formula[0]=='>':
    # Evaluate the operand.
    evaluated_operand = check_LTL(formula[1], trajectory, predicates)
    evaluation = [i for i in evaluated_operand[1:]]

  elif formula[0]=='%':
    # Evaluate the operands.
    evaluated_operand1 = check_LTL(formula[1], trajectory, predicates)
    evaluated_operand2 = check_LTL(formula[2], trajectory, predicates)
    min_len_operand = min(len(evaluated_operand1), len(evaluated_operand2))
    i=0
    while(i<min_len_operand):
      try:
        op2_idx = evaluated_operand2[i:min_len_operand].index(True) + i # get the first time op2 is True
        evaluation += [all(evaluated_operand1[j:op2_idx]) for j in range(i,op2_idx)] + [True]
        i = op2_idx + 1
      except:
        evaluation += [False for _ in range(i, min_len_operand)] # op2 is never True
        break

  elif formula[0]=='->':
    # Evaluate the operands.
    evaluated_operand1 = check_LTL(formula[1], trajectory, predicates)
    evaluated_operand2 = check_LTL(formula[2], trajectory, predicates)
    min_len_operand = min(len(evaluated_operand1), len(evaluated_operand2))

    evaluation = (1-np.array(evaluated_operand1)).astype(bool)[:min_len_operand]
    try:
      evaluation += np.array(evaluated_operand1)[:min_len_operand]*np.array(evaluated_operand2)[:min_len_operand]
    except:
      pass
      # print(trajectory)
      # print(evaluated_operand1, evaluated_operand2)
    evaluation = list(evaluation)

  else:
    print("Unknown operator")
  
  # end of evaluation
  
  return evaluation


# -----------------------------------------------------------------------------
# using PLY To parse the following simple grammar.
#
#   expression : term DISJUNCTION term
#              | term CONJUNCTION term
#              | term
#
#   term       : term IMPLIES term
#              | term UNTIL term
#              | ALWAYS factor
#              | EVENTUALLY factor
#              | NEXT factor
#              | NEGATE factor
#              | factor
#
#   factor     : AP
#              | ALWAYS factor
#              | EVENTUALLY factor
#              | NEXT factor
#              | NEGATE factor
#              | LPAREN expression RPAREN
#
# -----------------------------------------------------------------------------

# --- Tokenizer

# All tokens must be named in advance.

tokens = ('ALWAYS', 'EVENTUALLY', 'UNTIL', 'IMPLIES', 'NEXT', 'NEGATE',
          'DISJUNCTION', 'CONJUNCTION', 'LPAREN', 'RPAREN','AP')

# Ignored characters
t_ignore = ' \t'

# Token matching rules are written as regexs
t_ALWAYS = r'\[\]'
t_EVENTUALLY = r'\<\>'
t_UNTIL = r'%'
t_IMPLIES = r'-\>'
t_NEXT = r'\>'
t_NEGATE = r'~'
t_DISJUNCTION = r'/\\'
t_CONJUNCTION = r'\\/'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_AP = r'[a-zA-Z_][a-zA-Z0-9_]*'

# Ignored token with an action associated with it
def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')

# Error handler for illegal characters
def t_error(t):
    print(f'Illegal character {t.value[0]!r}')
    t.lexer.skip(1)

# Build the lexer object
lexer = lex()
    
# --- Parser

# Write functions for each grammar rule which is
# specified in the docstring.
def p_expression(p):
    '''
    expression :   term DISJUNCTION term
                 | term CONJUNCTION term
    '''
    # p is a sequence that represents rule contents.

    p[0] = (p[2], p[1], p[3])

def p_expression_term(p):
    '''
    expression : term
    '''
    p[0] = p[1]

def p_term(p):
    '''
    term : ALWAYS factor
        | EVENTUALLY factor
        | NEXT factor
        | NEGATE factor
    '''
    p[0] = (p[1], p[2])

def p_term_binary(p):
    '''
       term : term IMPLIES term
            | term UNTIL term
            | term DISJUNCTION term
            | term CONJUNCTION term
    '''
    p[0] = (p[2], p[1], p[3])

def p_term_factor(p):
    '''
    term : factor
    '''
    p[0] = p[1]

def p_factor_name(p):
    '''
    factor : AP
    '''
    p[0] = (None, p[1])

def p_factor_unary(p):
    '''
    factor : ALWAYS factor
           | EVENTUALLY factor
           | NEXT factor
           | NEGATE factor
    '''
    p[0] = (p[1], p[2])


def p_factor_grouped(p):
    '''
    factor : LPAREN expression RPAREN
    '''
    p[0] = p[2]

def p_error(p):
    print(f'Syntax error at {p.value!r}')

# Build the parser
parser = yacc()

# # Parse an expression

t = "[] ( (~d) /\ ((b /\ ~ > b) -> >(~b % (a \/ c))) /\ (a -> >(~a % b)) /\ ((~b /\ >b /\ ~>>b)->(~a % c)) /\ (c->(~a % b)) /\ ((b /\>b)-><>a))"

# ast = parser.parse(t)
# print(ast)