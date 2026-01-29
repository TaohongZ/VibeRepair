def build_d4j1_2():
    bugs = []

    # Chart:
    for i in range(1, 27):
        bugs.append('Chart-{}'.format(i))

    # Closure
    for i in range(1, 134):
        if i != 63 and i != 93:
            bugs.append('Closure-{}'.format(i))

    # Lang
    for i in range(1, 66):
        if i != 2 and i != 18 and i != 25 and i != 48:
            bugs.append('Lang-{}'.format(i))

    # Math
    for i in range(1, 107):
        bugs.append('Math-{}'.format(i))

    # Mockito
    for i in range(1, 39):
        bugs.append('Mockito-{}'.format(i))

    # Time
    for i in range(1, 28):
        if i != 21:
            bugs.append('Time-{}'.format(i))

    return bugs