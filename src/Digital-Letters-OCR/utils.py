import torch


class strLabelConverter(object):
    def __init__(self, chars, ignore_case):
        self.chars = chars
        self.ignore_case = ignore_case

        if ignore_case:
            self.chars = self.chars.lower() + '-' # 大写转小写后追加 '-' 作 blank 视觉占位

        else:
            self.chars = chars + '-' # 追加 '-' 作 blank 视觉占位:decode 用 chars[t[i]-1] 反查,当 t[i]=0(CTC blank) 时 i-1=-1 取 chars 末位即 '-',靠 Python 负索引天然完成 blank 归位,无需 if/else

        self.dict = {}

        for i, char in enumerate(self.chars):
            self.dict[char] = i + 1 # 0 留给 CTC blank;1..N 对应 chars[0..N-1];decode(idx) 用 chars[idx-1]

    def encode(self, text):
        """
        目的是将字符映射到对应的索引编号
        """

        if isinstance(text, str):
            text = [
                self.dict.get(char.lower() if self.ignore_case else char) 
                for char in text
            ]
            length = [len(text)]

        elif isinstance(text, list): # collection.Iterable 会报错，所以这里用list

            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, f"text with length {t.numel()} does not match declared length {length}"

            if raw:
                return ''.join([self.chars[i-1] for i in t])
            
            else:
                char_list = []
                for i in range(length):
                    if t[i] !=0 and (not (i>0 and t[i-1] == t[i])):
                        char_list.append(self.chars[t[i]-1])
                return ''.join(char_list)

        else:
            assert t.numel() == length.numel(), f"text with length {t.numel()} does not match declared length {length.numel()}"
            texts = []
            idx = 0

            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[idx:idx+l], torch.IntTensor([l]), raw=raw)
                    )
                idx += l
        

