"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""
# from . import cmudict

_pad        = "_"
_eos        = "~"
_characters_en = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "
_characters_lt = "AaĄąBbCcČčDdEeĘęĖėFfGgHhIiĮįYyJjKkLlMmNnOoPpRrSsŠšTtUuŲųŪūVvZzŽž!\'\"(),-.:;? "
_phonemes_lt = "\u0020\u0021\u0028\u0029\u002c\u002e\u003a\u003b\u003f\u005b\u005d\u0061\u0062\u0064\u0065\u0066\u0068\u0069\u006a\u006b\u006c\u006d\u006e\u006f\u0070\u0072\u0073\u0074\u0075\u0076\u0078\u007a\u0250\u0251\u0254\u0255\u025b\u0261\u026a\u026d\u0282\u0283\u028a\u0291\u0292\u02b2\u02d0\u0329"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters_lt) #+ _arpabet
# symbols = [_pad, _eos] + list(_phonemes_lt)