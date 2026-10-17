import os, json, random, zipfile, hashlib, platform, importlib.metadata, uuid
from pathlib import Path
from datetime import datetime, timezone
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from IPython.display import display
SEEDS = (101, 202, 303, 404, 505)
tf.config.experimental.enable_op_determinism()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
OUT = Path("ysa_stability") / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
print("Çalışma:", RUN_ID, "TensorFlow:", tf.__version__)


from google.colab import files
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Tek bir ana deney arşivi gereklidir.")
source_name, source_bytes = next(iter(uploaded.items()))
import io
with zipfile.ZipFile(io.BytesIO(source_bytes)) as z:
    def read_member(basename):
        matches = [n for n in z.namelist() if not n.endswith("/") and Path(n).name == basename]
        if len(matches) != 1:
            raise ValueError(f"{basename}: tam bir eşleşme bekleniyordu; bulunan {len(matches)}")
        return z.read(matches[0])
    split_bytes = read_member("split_indices.npz")
    split = np.load(io.BytesIO(split_bytes), allow_pickle=False)
    idx_train, idx_val, idx_test = [split[k].copy() for k in ("train", "validation", "test")]
    original = pd.read_csv(io.BytesIO(read_member("table_6_final_performance.csv")))
    original_neural = pd.read_csv(io.BytesIO(read_member("table_4_neural_experiments.csv")))
    original_run = json.loads(read_member("run_summary.json"))
    original_env = json.loads(read_member("environment.json"))
    archived_cnn = read_member("neural_2_best.keras")
    with zipfile.ZipFile(io.BytesIO(archived_cnn)) as model_zip:
        original_cnn_config = json.loads(model_zip.read("config.json"))
(OUT / "split_indices.npz").write_bytes(split_bytes)
original.to_csv(OUT / "original_performance.csv", index=False)
(OUT / "original_environment.json").write_text(json.dumps(original_env, indent=2))
digits = load_digits()
y = digits.target.astype("int64")
for idx in (idx_train, idx_val, idx_test):
    assert idx.ndim == 1 and np.issubdtype(idx.dtype, np.integer)
    assert idx.min() >= 0 and idx.max() < len(y)
assert (len(idx_train), len(idx_val), len(idx_test)) == (1257,270,270)
all_idx = np.concatenate([idx_train, idx_val, idx_test])
assert len(all_idx) == len(y)
assert len(np.unique(all_idx)) == len(y)
assert np.array_equal(np.sort(all_idx), np.arange(len(y)))
assert original_run["dataset"] == "scikit-learn Digits"
assert original_run["selected_neural_model"] == "CNN-C Görüntü modeli"
row = original_neural.loc[original_neural["Mimari"].eq("CNN")].iloc[0]
assert np.isclose(row.Dropout,.30) and np.isclose(row.L2,1e-4)
assert np.isclose(row.Learning_Rate,1e-3) and row.Batch == 32
assert int(row.Param_Count) == 33002
x_train = (digits.images[idx_train].astype("float32") / 16.0)[...,None]
x_val = (digits.images[idx_val].astype("float32") / 16.0)[...,None]
y_train, y_val = y[idx_train], y[idx_val]
versions = {n: importlib.metadata.version(n) for n in ["tensorflow","keras","numpy","pandas","scikit-learn","scipy","matplotlib"]}
versions["python"] = platform.python_version()
(OUT / "environment.json").write_text(json.dumps(versions,indent=2))
changes = {k: [original_env.get(k),v] for k,v in versions.items() if original_env.get(k) != v}
print("Önceki ortama göre sürüm farkları:", changes or "Yok")
plan = {"seeds": list(SEEDS), "epochs_max":250, "batch_size":32,
        "architecture":"CNN-C", "dropout":.30, "l2":1e-4, "learning_rate":1e-3,
        "source_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "split_sha256":hashlib.sha256(split_bytes).hexdigest(),
        "planned_after_original_test_review":True,
        "archived_reference_model_sha256":hashlib.sha256(archived_cnn).hexdigest(),
        "selection_rule":"Each run: minimum validation loss. No test-based run selection.",
        "scope":"Training randomness conditional on one fixed split; not independent test replications."}
(OUT / "experiment_plan.json").write_text(json.dumps(plan,indent=2))
print("Veri ve mimari doğrulandı. Plan kaydedildi. Tohumlar:",SEEDS)


def build_cnn(dropout, l2_value, learning_rate):
    model = models.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2_value)),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2_value)),
        layers.MaxPooling2D(2),
        layers.Dropout(dropout),
        layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2_value)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_value)),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
def architecture_signature(config):
    items = []
    for layer in config["config"]["layers"]:
        cls, params = layer["class_name"], layer.get("config", {})
        if cls == "InputLayer":
            continue
        keys = ("filters", "kernel_size", "padding", "activation", "units", "rate", "axis", "pool_size")
        items.append((cls, {k: params[k] for k in keys if k in params}))
    return items

reference_signature = architecture_signature(original_cnn_config)
model = build_cnn(.30, 1e-4, 1e-3)
current_signature = architecture_signature(json.loads(model.to_json()))
assert current_signature == reference_signature, (
    "CNN katmanları ana deneydeki CNN-C ile eşleşmiyor. Eğitim durduruldu.")
assert model.count_params() == 33002, f"Parametre sayısı beklenenden farklı: {model.count_params()}"
print("Ana deneydeki CNN-C katmanları ve parametre sayısı doğrulandı.")


training_rows = []
for seed in SEEDS:
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    model = build_cnn(.30,1e-4,1e-3)
    assert model.count_params() == 33002
    checkpoint = OUT / f"cnn_seed_{seed}.keras"
    cb = [callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min", save_best_only=True),
          callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=18, min_delta=1e-4, restore_best_weights=True),
          callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=.5, patience=7, min_lr=1e-6)]
    print(f"Başlıyor: seed={seed}", flush=True)
    history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=250,
                        batch_size=32,callbacks=cb,shuffle=True,verbose=2)
    h = {k:[float(v) for v in values] for k,values in history.history.items()}
    (OUT / f"history_seed_{seed}.json").write_text(json.dumps(h))
    model = models.load_model(checkpoint)
    prob = model.predict(x_val,verbose=0)
    training_rows.append({"seed":seed,"epochs_run":len(h["loss"]),
        "best_epoch":int(np.argmin(h["val_loss"])+1),"min_val_loss":float(min(h["val_loss"])),
        "validation_accuracy":accuracy_score(y_val,prob.argmax(1)),
        "validation_macro_f1":f1_score(y_val,prob.argmax(1),average="macro")})
    pd.DataFrame(training_rows).to_csv(OUT / "training_runs.csv",index=False)
    del model
assert len(training_rows) == len(SEEDS)
print("Beş eğitim de tamamlandı; şimdi test değerlendirmesi yapılabilir.")
display(pd.DataFrame(training_rows))


assert len(training_rows) == len(SEEDS)
x_test = (digits.images[idx_test].astype("float32") / 16.0)[...,None]
y_test = y[idx_test]
def measure(y_true, prob):
    pred = prob.argmax(1)
    return {"test_accuracy":accuracy_score(y_true,pred),
            "test_macro_f1":f1_score(y_true,pred,average="macro"),
            "test_macro_precision":precision_score(y_true,pred,average="macro",zero_division=0),
            "test_macro_recall":recall_score(y_true,pred,average="macro",zero_division=0),
            "test_macro_auc":roc_auc_score(y_true,prob,multi_class="ovr",average="macro"),
            "correct":int((pred==y_true).sum()), "errors":int((pred!=y_true).sum())}
rows=[]
for seed in SEEDS:
    tf.keras.backend.clear_session()
    model=models.load_model(OUT / f"cnn_seed_{seed}.keras")
    prob=model.predict(x_test,verbose=0)
    assert prob.shape == (270,10) and np.isfinite(prob).all()
    np.save(OUT / f"test_probabilities_seed_{seed}.npy",prob)
    pd.DataFrame({"sample_index":idx_test,"true_label":y_test,"prediction":prob.argmax(1)}).to_csv(
        OUT / f"test_predictions_seed_{seed}.csv",index=False)
    rows.append({"seed":seed,**measure(y_test,prob)})
results=pd.DataFrame(training_rows).merge(pd.DataFrame(rows),on="seed",validate="one_to_one")
assert list(results.seed) == list(SEEDS)
results.to_csv(OUT / "all_five_runs.csv",index=False)
metric_cols=[c for c in results if c.startswith("test_") or c.startswith("validation_")]
def summarize(frame,columns):
    return frame[columns].agg(["mean","std","min","max"]).T.rename_axis("metric").reset_index()
summary=summarize(results,metric_cols) # pandas std uses ddof=1 (sample SD).
summary.to_csv(OUT / "stability_summary.csv",index=False)
display(results.round(5));display(summary.round(5))
print("Standart sapma örneklem standart sapmasıdır (ddof=1). Aynı test kümesi beş kez kullanılmıştır.")


fig, axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,col,original_col,title in zip(axes,["test_accuracy","test_macro_f1"],
        ["Test_Accuracy","Test_Macro_F1"],["Test doğruluğu","Test makro F1"]):
    ax.plot(results.seed.astype(str),results[col],"o-",label="Beş CNN koşusu")
    svm_rows=original.loc[original.Model.eq("SVM (RBF)"),original_col]
    assert len(svm_rows) == 1
    svm=float(svm_rows.iloc[0])
    ax.axhline(svm,ls="--",color="gray",label="Ana deney SVM (tek koşu)")
    ax.set(xlabel="Başlangıç tohumu",ylabel="Skor",title=title)
    ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(OUT / "stability_five_runs.png",dpi=600,bbox_inches="tight");plt.show()
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for seed in SEEDS:
    h=json.loads((OUT / f"history_seed_{seed}.json").read_text())
    axes[0].plot(np.arange(1,len(h["val_loss"])+1),h["val_loss"],label=str(seed))
    axes[1].plot(np.arange(1,len(h["val_accuracy"])+1),h["val_accuracy"],label=str(seed))
for ax,title in zip(axes,["Doğrulama kaybı","Doğrulama doğruluğu"]):
    ax.set(xlabel="Epoch",title=title);ax.legend(title="Seed");ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT / "validation_curves_five_runs.png",dpi=600,bbox_inches="tight");plt.show()
a=summary.set_index("metric").loc["test_accuracy"]
f=summary.set_index("metric").loc["test_macro_f1"]
text=(f"Sabit veri ayrımında, önceden belirlenmiş beş başlangıç tohumu ile CNN-C eğitilmiştir. "
      f"Test doğruluğu ortalaması %{100*a['mean']:.2f}, örneklem standart sapması {100*a['std']:.2f} yüzde puandır. "
      f"Doğruluk aralığı %{100*a['min']:.2f}–%{100*a['max']:.2f}; makro F1 ortalaması {f['mean']:.5f}, "
      f"standart sapması {f['std']:.5f} bulunmuştur. "
      "Bu değişkenlik yalnızca aynı veri bölünmesindeki eğitim rastgeleliğini yansıtır; "
      "veri bölünmesinden veya yeni örneklemden kaynaklanan belirsizliği ölçmez. "
      "Analiz ana deneyin test sonuçları görüldükten sonra planlanmıştır. Teste göre koşu seçilmemiştir. "
      "SVM karşısında anlamlı üstünlük veya eşdeğerlik iddiası için bu özet tek başına yeterli değildir.")
(OUT / "analysis_note.txt").write_text(text,encoding="utf-8")
print(text)


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

per_class_rows = []
all_predictions = []
for seed in SEEDS:
    pred_df = pd.read_csv(OUT / f"test_predictions_seed_{seed}.csv")
    assert np.array_equal(pred_df.sample_index.to_numpy(), idx_test)
    assert np.array_equal(pred_df.true_label.to_numpy(), y_test)
    pred = pred_df.prediction.to_numpy()
    assert np.isin(pred, np.arange(10)).all()
    all_predictions.append(pred)
    report = classification_report(y_test, pred, labels=list(range(10)), output_dict=True, zero_division=0)
    for label in range(10):
        values = report[str(label)]
        per_class_rows.append({"seed":seed,"class":label,
            "precision":values["precision"],"recall":values["recall"],
            "f1":values["f1-score"],"support":int(values["support"])})
    matrix = confusion_matrix(y_test,pred,labels=list(range(10)))
    pd.DataFrame(matrix,index=range(10),columns=range(10)).to_csv(OUT / f"confusion_matrix_seed_{seed}.csv")
    fig, ax = plt.subplots(figsize=(6,5))
    img = ax.imshow(matrix,cmap="Blues")
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=7,
                    color="white" if matrix[i,j] > matrix.max()/2 else "black")
    ax.set(xticks=range(10),yticks=range(10),xlabel="Tahmin",ylabel="Gerçek",
           title=f"CNN-C — Seed {seed} test karmaşıklık matrisi")
    fig.colorbar(img,ax=ax,fraction=.045)
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_matrix_seed_{seed}.png",dpi=180,bbox_inches="tight")
    plt.close(fig)

pd.DataFrame(per_class_rows).to_csv(OUT / "per_class_metrics.csv",index=False)
all_predictions = np.stack(all_predictions)
consensus = np.array([np.bincount(all_predictions[:,i],minlength=10).argmax() for i in range(len(y_test))])
variety = np.array([len(np.unique(all_predictions[:,i])) for i in range(len(y_test))])
pd.DataFrame({"sample_index":idx_test,"true_label":y_test,
              "correct_seed_count":(all_predictions==y_test).sum(axis=0),
              "distinct_prediction_count":variety,
              "all_seeds_agree":variety==1,
              "majority_prediction_descriptive_only":consensus}).to_csv(
                  OUT / "per_sample_stability.csv",index=False)

pair_rows=[]
for i,s1 in enumerate(SEEDS):
    for j,s2 in enumerate(SEEDS):
        if j<=i: continue
        pair_rows.append({"seed_1":s1,"seed_2":s2,
                          "prediction_agreement":float((all_predictions[i]==all_predictions[j]).mean()),
                          "cohen_kappa":float(cohen_kappa_score(all_predictions[i],all_predictions[j]))})
pd.DataFrame(pair_rows).to_csv(OUT / "pairwise_prediction_agreement.csv",index=False)
print(f"Beş seed'in tamamının aynı sınıfı tahmin ettiği test örneği: {(variety==1).sum()}/270")
print("Çoğunluk tahmini yalnızca açıklayıcıdır; önceden belirlenen beş koşunun performansının yerine kullanılmaz.")


from shutil import make_archive
required = ["experiment_plan.json", "split_indices.npz", "environment.json",
            "all_five_runs.csv", "stability_summary.csv", "training_runs.csv",
            "per_class_metrics.csv", "per_sample_stability.csv",
            "pairwise_prediction_agreement.csv", "analysis_note.txt",
            "stability_five_runs.png", "validation_curves_five_runs.png"]
for seed in SEEDS:
    required += [f"cnn_seed_{seed}.keras", f"history_seed_{seed}.json",
                 f"test_probabilities_seed_{seed}.npy",
                 f"test_predictions_seed_{seed}.csv",
                 f"confusion_matrix_seed_{seed}.csv", f"confusion_matrix_seed_{seed}.png"]
missing = [name for name in required if not (OUT/name).is_file() or (OUT/name).stat().st_size==0]
assert not missing, f"Eksik veya boş sonuç dosyaları: {missing}"
assert list(pd.read_csv(OUT/'all_five_runs.csv')['seed']) == list(SEEDS)
for seed in SEEDS:
    assert np.load(OUT/f"test_probabilities_seed_{seed}.npy").shape==(270,10)
manifest={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in sorted(OUT.iterdir()) if p.is_file()}
(OUT/'file_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
archive=make_archive(f"YSA_F26_Kararlilik_{RUN_ID}","zip",root_dir=OUT)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, "ZIP bütünlük testi başarısız"
    assert set(required).issubset(set(z.namelist()))
print("Dosya sayısı:",len(manifest),"ZIP bütünlük testi: başarılı")
print("Tamamlandı:",archive)
files.download(archive)
