
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import sqlite3


def load_model(model, load_model_db, model_rowid=None):
    con = sqlite3.connect(load_model_db, timeout=60)
    cursor = con.cursor()
    try:
        if model_rowid is not None:
            rec = list(
                cursor.execute(
                    f"SELECT rowid, * FROM Models where rowid = {model_rowid}"
                )
            )
        else:
            rec = list(
                cursor.execute(
                    "SELECT rowid, * FROM Models ORDER BY rowid DESC LIMIT 1"
                )
            )
    except sqlite3.OperationalError:
        print("Failed to load model from database.")
        return
    finally:
        con.close()

    rowid, config, kwargs, state_dict, state_dict_ema = rec[0]

    state_dict = pickle.loads(state_dict)
    state_dict_ema = pickle.loads(state_dict_ema)
    if state_dict_ema is not None:
        state_dict = state_dict_ema

    msg = model.load_state_dict(state_dict)

    print(f"Initialized model with the checkpoint from database {load_model_db} row {rowid}: {msg}")