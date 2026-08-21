from Dassl.dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry("TRAINER")

from trainers.CLIP import CLIP
from trainers.GLP_OT import GLP_OT
from trainers.FEDPGP import FEDPGP
from trainers.PROMPTFL import PROMPTFL
from trainers.GL_SVDMSE import GL_SVDMSE
from trainers.GL_SVDMSE_HE import GL_SVDMSE_HE
from trainers.GL_SVDMSE_VSP import GL_SVDMSE_VSP
# from trainers.GL_SVDMSE_CGL import GL_SVDMSE_CGL  # imported via federated_main.py to avoid circular import
# from trainers.GL_SVDMSE_CGLA import GL_SVDMSE_CGLA  # imported via federated_main.py
# from trainers.GL_SVDMSE_CGLU import GL_SVDMSE_CGLU  # imported via federated_main.py

TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(GLP_OT)
TRAINER_REGISTRY.register(FEDPGP)
TRAINER_REGISTRY.register(PROMPTFL)
TRAINER_REGISTRY.register(GL_SVDMSE)
TRAINER_REGISTRY.register(GL_SVDMSE_HE)
TRAINER_REGISTRY.register(GL_SVDMSE_VSP)
# GL_SVDMSE_CGL is registered via @TRAINER_REGISTRY.register() in GL_SVDMSE_CGL.py

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
