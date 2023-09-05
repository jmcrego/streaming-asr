import logging

class Hyp():

    def __init__(self, response_json, start, end):
        """ response_json is like: {'transcript': {'hyp': [[0.0, 0.28, ' Ouf'], [0.28, 0.54, ' !']], 'language': 'fr', 'language_probability': 1}} """
        self.start = start
        self.end = end
        self.language = None
        self.language_probability = None
        self.hyp = None
        if 'transcript' in response_json:
            t = response_json['transcript']
            if 'language' in t and 'language_probability' in t and 'hyp' in t:
                self.language = t['language']
                self.language_probability = t['language_probability']
                self.hyp = t['hyp']
                if len(self.hyp):
                    print("hyp: {}".format(str(self)), end='\n' if logging.root.level == logging.INFO else '\r')

    def __len__(self):
        return len(self.hyp) if self.hyp is not None else 0

    def __str__(self):
        return ''.join([t[2] for t in self.hyp]).strip()
       
    def has_endchars(self, end_chars, skip_ini, skip_end):
        """ returns the position of the FIRST recognised token that is ended by an ending char
        Params:
        end_chars: a string containing the valid end chars (. ? ! ,)
        skip_ini: The n initial tokens are not considered
        skip_end: The n final tokens are not considered
        Returns:
        n: the position of the token ended by one of the chars in end_chars or None
        """
        if self.hyp is None:
            return None
        for n in range(skip_ini, len(self.hyp)-skip_end):
            t = self.hyp[n]
            if not t[2].endswith('...') and ( t[2].endswith(tuple(list(end_chars))) or t[2].startswith(tuple(list(end_chars))) ):
                return n
        return None

    def remove_after_token_n(self, n, sample_rate):
        assert len(self.hyp) > n+1
        self.end = self.start + sample_rate * int(self.hyp[n][1]+self.hyp[n+1][0])//2
        self.hyp = self.hyp[:n+1]
