import skimage._shared.lazy as lazy


def test_attach():
    name = "mymod"
    submods = ["mysubmodule", "anothersubmodule"]
    myall = {"not_real_submod": ["some_var_or_func"]}

    locls = {
        "attach": lazy.attach,
        "name": name,
        "submods": submods,
        "myall": myall,
    }
    s = "__getattr__, __lazy_dir__, __all__ = attach(name, submods, myall)"

    exec(s, {}, locls)
    expected = {
        "attach": lazy.attach,
        "name": name,
        "submods": submods,
        "myall": myall,
        "__getattr__": None,
        "__lazy_dir__": None,
        "__all__": None,
    }
    assert locls.keys() == expected.keys()
    for k, v in locls.items():
        if v is not None:
            assert expected[k] == v
