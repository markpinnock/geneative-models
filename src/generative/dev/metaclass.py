from typing import Any

obj_dict = {}
cls_dict = {}
metacls_dict = {}


class CustomDict(dict):
    def __setitem__(self, key, value):
        print(f"setting {key} to {value}")
        super().__setitem__(key, value)


print("Declaring MetaClass\n")


class MetaClass(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        print(
            f"MetaClass.__prepare__: mcs={metacls}, name={name}, bases={bases}, kwds={kwds}",
        )
        return CustomDict()

    def __new__(metacls, name, bases, attrs, **kwds):
        print(
            f"MetaClass.__new__: mcls={metacls}, name={name}, bases={bases}, attrs={attrs}, kwds={kwds}\n",
        )
        metacls_dict[metacls.__name__] = metacls
        cls = super().__new__(metacls, name, bases, attrs)
        cls_dict[cls.__name__] = cls
        return cls

    def __init__(cls, name, bases, attrs, **kwds):
        print(
            f"MetaClass.__init__: cls={cls}, name={name}, bases={bases}, attrs={attrs}, kwds={kwds}\n",
        )
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwds):
        print(f"MetaClass.__call__: args {args}, kwds {kwds}\n")
        return super().__call__(*args, **kwds)


print(MetaClass, type(MetaClass), "\n")
print("Declaring MyClass\n")


class MyClass(metaclass=MetaClass, extra=1):
    def __new__(cls, **kwds):
        print(f"MyClass.__new__: cls={cls}, kws={kwds}\n")
        instance = super().__new__(cls)
        obj_dict["object"] = instance
        return instance

    def __init__(self, **kwds):
        print(f"MyClass.__init__: kwds={kwds}\n")
        super().__init__()

    def __call__(self, *args):
        print(f"MyClass __call__: args {args}\n")


print(MyClass, type(MyClass), "\n")

print("Initialising MyClass\n")
mc = MyClass(a=2, b=3)
print(mc, type(mc), "\n")

print("Calling MyClass\n")
mc(3, 4, 5)

print(metacls_dict)
print(cls_dict)
print(obj_dict)
