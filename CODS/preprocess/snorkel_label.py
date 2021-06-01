'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from snorkel.labeling.model import LabelModel
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
import pandas as pd
import re

#from dataset import load_dialogue

WHY = 0
WHAT = 1
WHERE = 2
WHEN = 3
CONFIRM = 4
ABSTAIN = -1


@labeling_function()
def lf_why_keyword(x):
    matches = ['why ', 'y ?', 'whys ', ' y.', ' y .']
    not_matches = ['why not', 'that\'s why', 'I can see why']
    # return WHY if any(item in x.text.lower() for item in matches) else ABSTAIN
    return WHY if (any(item in x.text.lower() for item in matches) and
                   all(item not in x.text.lower() for item in not_matches))else ABSTAIN


@labeling_function()
def lf_what_keyword(x):
    matches = ['what', 'what\'s up', 'what about', 'how']
    return WHAT if (any(item in x.text.lower() for item in matches) and
                    '?' in x.text.lower())else ABSTAIN


@labeling_function()
def lf_where_keyword(x):
    return WHERE if ("where" in x.text.lower() and
                     '?' in x.text.lower()) else ABSTAIN


@labeling_function()
def lf_when_keyword(x):
    matches = ['when']
    soft_matches = ['what time']
    return WHEN if ((any(item in x.text.lower() for item in matches) and
                    '?' in x.text.lower())
                    or (any(item in x.text.lower() for item in soft_matches))) else ABSTAIN

@labeling_function()
def lf_confirm_keyword(x):
    matches = ['are you', 'do you', 'did you', 'can you', 'could you', 'could u', 'have you', 'will you', 'did anyone',
               'can we', 'can I', 'is she', 'is he', 'has she', 'has he', 'has anyone']
    not_matches = ['where ', 'who ', 'when ', 'why ', 'what ', 'how ']
    return CONFIRM if (any(item in x.text.lower() for item in matches) and
                       all(item not in x.text.lower() for item in not_matches)) else ABSTAIN


def filter_function_dialog(dialogs):
    dialog_text = []
    for dialog in dialogs:
        dialog_text += dialog['function_dialogs']
    return dialog_text


def dispatch_labels(train_data, dialogs):
    """ Dispatch snorkel labels for each dialog"""
    start_idx = 0
    for dialog in dialogs:
        num_function_dialogs = len(dialog['function_dialogs'])
        end_idx = start_idx + num_function_dialogs
        labels = train_data.label[start_idx:end_idx].tolist()
        dialog['module_index'] = labels
        start_idx += num_function_dialogs
    return dialogs


def get_snorkel_label(train_dialogs, eval_dialogs, test_dialogs):
    # Use Train data to train the label_model
    func_dialogs = filter_function_dialog(train_dialogs)
    train_data = pd.DataFrame(func_dialogs, columns=['text'])
    lfs = [lf_why_keyword, lf_what_keyword, lf_where_keyword, lf_when_keyword, lf_confirm_keyword]
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(train_data)

    # Train a model
    label_model = LabelModel(cardinality=6, verbose=True)
    label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
    train_data['label'] = label_model.predict(L=L_train, tie_break_policy="abstain")
    snorkel_labeled_train_dialogs = dispatch_labels(train_data, train_dialogs)

    # Label the eval dialogs
    func_dialogs = filter_function_dialog(eval_dialogs)
    eval_data = pd.DataFrame(func_dialogs, columns=['text'])
    L_eval = applier.apply(eval_data)
    eval_data['label'] = label_model.predict(L=L_eval, tie_break_policy="abstain")
    snorkel_labeled_eval_dialogs = dispatch_labels(eval_data, eval_dialogs)

    # Label the test dialogs
    func_dialogs = filter_function_dialog(test_dialogs)
    test_data = pd.DataFrame(func_dialogs, columns=['text'])
    L_test = applier.apply(test_data)
    test_data['label'] = label_model.predict(L=L_test, tie_break_policy="abstain")
    snorkel_labeled_test_dialogs = dispatch_labels(test_data, test_dialogs)

    return snorkel_labeled_train_dialogs, snorkel_labeled_eval_dialogs, snorkel_labeled_test_dialogs


# def get_snorkel_model(dialogs):
#     func_dialogs = filter_function_dialog(dialogs)
#     train_data = pd.DataFrame(func_dialogs, columns=['text'])
#     lfs = [lf_why_keyword, lf_what_keyword, lf_where_keyword, lf_when_keyword, lf_number_keyword, lf_confirm_keyword]
#     applier = PandasLFApplier(lfs)
#     L_train = applier.apply(train_data)
#
#     # Train a model
#     label_model = LabelModel(cardinality=6, verbose=True)
#     label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
#     train_data['label'] = label_model.predict(L=L_train, tie_break_policy="abstain")
#     return train_data



