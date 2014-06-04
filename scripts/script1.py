#====================================================================
# purpose:
# author:
# created:
# revised:
# dependencies:
# comments:
#====================================================================

import html

# writing a function that accepts an arbitrary number of arguments
def make_element(name, value, **attrs):
    keyvals = ['%s = "%s"' % item for item in attrs.items()]
    attr_str = ' '.join(keyvals)
    element = '<{name} {attrs}>{value}</{name}>'.format(
                                                name = name,
                                                attrs = attr_str,
                                                value = html.escape(value))
    return(element)

make_element('p', '<spam>')
make_element('item', 'Albatross', size = 'Large', quantity = 6)

# returning multiple values: Python 3 syntax
def fnFoo():
    return(1, 2, 3)

isinstance(fnFoo(), tuple)