"""Static action sets for binary to discrete action space wrappers."""

""" I remove orther action just keep the thing i need """



# actions for more complex movement / only thing i need
COMPLEX_MOVEMENT = [
    ['right'],
    ['NOOP'],
    
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]
