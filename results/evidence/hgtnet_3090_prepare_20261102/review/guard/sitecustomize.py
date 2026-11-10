import builtins
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:
        raise AssertionError('ML import forbidden in independent inspect/status: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
