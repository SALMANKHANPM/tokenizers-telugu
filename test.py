from indicnlp import loader  
loader.load()  
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator  
  
from ai4bharat.transliteration import XlitEngine
  
# Initializing the en-indic multilingual model and dictionaries (if rerank option is True)
e = XlitEngine("te", beam_width=4, rescore=True, src_script_type="latin")

# Transliterate word
out = e.translit_sentence("""sataru Di.vi.Di pratini si.ke.gaariki chEra vEyagaa vaari sateemaNi si.ke.laavaNya eTuvaMTi hipaakrasi lEka svayaMgaa phOn dvaaraa maaTlaaDi saMtOshaM vyaktaM chEsi prOtsa hiMchaDaM nijaMgaa garvakaaraNaM.
chittooru Sree kRshNaa jyuyalars vaari saujanyaMtO paajiTiv thiMkars klab vaaru sthaanika jaDpi meeTiMg haalulO nirvahiMchina seminaarlO nEnu chEsina prasaMgaM yokka saMpoorNa dRSyamaalikatO ee Di.vi.Di roopudiddukunnadi.
Dabbu lEni vaaru uMDochchugaani Dabbu avasaraM lEni vaaru ekkaDaa lEru. ayitE Dabbu guriMchina saraina avagaahaNa leka maanavulu naanaa taMTaalu paDutunnaaru. iMtakee Dabbuyokka Saktini maanavuDu atigaa oohiMchukOvaDaM, maraNaM, maraNaM yokka Chaayalaina oMTaritanaM,cheekaTi,tiraskaaraM, musalitanaM, OTami, jabbulanuMDi rakshistuMdani sab kaanshiyas gaa bhaaviMchaDaM valanE Dabbunu sariggaa arthaM chEsukOlEka pOyaaDu maanavuDu.
Dabbu guriMchina apOhalanu tolagiMchi,prativaarini saMpaadanaku poonukunElaa cheyyaDamE ee Di.vi.Di lOni prasaMgaM yokka dyEyaM.
aMdukE sataru Di.vi.Dini prati okkariki aMdubhaaTulO tEvaalani chakkagaa kaMpres chEsi mobail phOnlO saitaM DavunlOD chEsukunE vidaMgaa aan lain lO uMchaanu.
iMkaa aalasyaM eMduku ?""")
print(out)

# uv pip install "tensorflow==2.15.1" "keras==2.15.0" "tensorflow-addons==0.23.0" 
# requires python 3.10 for fairseq 

#    with open(local_path, "rb") as f:
#       state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)