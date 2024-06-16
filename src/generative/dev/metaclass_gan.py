import abc


class Metaclass(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        use_mixin = kwargs.pop("use_mixin", False)
        print(f"{__class__}.__call__:", cls.__name__)

        # Dynamically create a new class with Mixin and BaseClass
        if use_mixin:
            cls = type("SubClassWithMixin", (WassersteinMixin, cls), {})
            print(f"{__class__}.__call__:", cls.__name__)

        return super().__call__(*args, **kwargs)


class BaseGAN(metaclass=Metaclass):
    def __init__(self, *args, **kwargs):
        print(f"Entering {__class__}.__init__")
        super().__init__(*args, **kwargs)
        print(f"Leaving {__class__}.__init__")

    @abc.abstractmethod
    def other_func(self):
        pass

    def train_step(self):
        print("Normal training step")


class WassersteinMixin:
    def __init__(self, a=100, *args, **kwargs):
        self.a = a
        print(f"Entering {__class__}.__init__")
        super().__init__(
            *args,
            **kwargs,
        )  # Ensure parent classes' __init__ are called if any
        print(f"Leaving {__class__}.__init__")

    def train_step(self):
        print(f"Wasserstein training step with {self.a}")


class SubGAN(BaseGAN):
    def __init__(self, *args, **kwargs):
        print(f"Entering {__class__}.__init__")
        super().__init__(*args, **kwargs)
        print(f"Leaving {__class__}.__init__")

    def other_func(self):
        pass


d = SubGAN(use_mixin=False)
print(d.__class__.mro())
d.train_step()
print()

d = SubGAN(use_mixin=True, a=100)
print(d.__class__.mro())
d.train_step()
