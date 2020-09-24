import unicodedata
from transformers import BasicTokenizer
import six
is_py2 = six.PY2
if not is_py2:
    basestring = str
def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)

def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')
def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')
def stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token
def rematch(text, tokens,do_lower_case=True):
    """给出原始的text和tokenize后的tokens的映射关系
    """

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        if do_lower_case:
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))
    text = normalized_text.lower()

    token_mapping, offset = [], 0
    for token in tokens:
        if _is_special(token):
            token_mapping.append([])
        else:
            token = stem(token)
            # print(token)
            # print('===')
            # print(text[offset:])
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping