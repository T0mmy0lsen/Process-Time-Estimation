import util


class Files:

    _input_path = None
    _name = None
    _df = None

    def input_path(self, input_path=None):
        if input_path is not None:
            self._input_path = input_path
        return str(self._input_path)


    def name(self, name=None):
        if name is not None:
            self._name = name
        return str(self._name)


    def df(self, df=None):
        if df is not None:
            self._df = df
        return self._df


    def __init__(self, name='_Files', input_path='./input/source/'):
        self.name(name)
        self.input_path(input_path)
        util.load_files_to_dataframe(self)
