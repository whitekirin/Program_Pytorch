class validation_the_enter_data:
    def __init__(self) -> None:
        pass
    def validation_string(self, content, Comparison):
        if content == Comparison:
            return True
        else:
            return False

    def validation_type(self, enter, Type: type):
        if not isinstance(enter, Type):
            return False
        else:
            return True