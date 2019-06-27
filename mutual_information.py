import torch
import numpy as np

def get_mutual_information(preds_student, preds_teacher):
    preds_student_new = preds_student.unsqueeze(1)
    preds_teacher_new = preds_teacher.unsqueeze(1)
    bs, dummy, cls = preds_student_new.size()
    predJoint = torch.bmm(preds_student_new.view(bs, cls, dummy), preds_teacher_new)
    preds_student_repeated = preds_student.repeat(1, cls)
    preds_teacher_repeated = preds_teacher.t().repeat(cls, 1)
    log_term = torch.log(predJoint) - torch.log(preds_student_repeated) - torch.log(preds_teacher_repeated)
    mutual_info = torch.sum(predJoint*log_term)
    return mutual_info
