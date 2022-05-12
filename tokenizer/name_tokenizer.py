class NameTokenizer():
    def __init__(self, path, required_gender="男"):

        self.names = []
        self._tokens = []
        self._tokens += ['0', '\n']
        with open(path, mode='r', encoding="utf-8-sig") as f:
            for idx, line in enumerate(f.readlines()):

                line = line.strip()
                line = line.split(",")
                gender = line[1]

                if gender == required_gender or gender == "未知":
                    name = line[0]
                    self.names.append(name)
                    for char in name:
                        self._tokens.append(char)

        # 创建词典 token->id映射关系
        self._tokens = sorted(list(set(self._tokens)))
        self.token_id_dict = dict((c, i) for i, c in enumerate(self._tokens))
        self.id_token_dict = dict((i, c) for i, c in enumerate(self._tokens))
        self.vocab = len(self._tokens)
