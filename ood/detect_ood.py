import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

def add_legend():
    plt.scatter([],[],s=15,c="blue",edgecolors="black", linewidths=.75, label="Train")
    plt.scatter([],[],s=15,c="red",edgecolors="black", linewidths=.75, label="Test")
    plt.legend()

def plot_recon_pca(pca, data):
    def individual_subplot(i, image, title):
        cbar_shrink = 1
        cbar_pad = 0.04
        plt.subplot(1,3,i)
        plt.imshow(image)
        plt.colorbar(location="bottom", shrink=cbar_shrink, pad=cbar_pad)
        plt.axis("equal")
        plt.axis("off")
        plt.title(title)

    output_np = pca.inverse_transform(pca.transform(data.reshape(1,-1))).reshape(64,64)
    err = np.abs(data - output_np)

    plt.figure(figsize=(12,4.5),dpi=110)
    individual_subplot(1, data, "Input")
    individual_subplot(2, output_np, "PCA Reconstruction")
    individual_subplot(3, err, "Error")
    plt.show()


if __name__ == "__main__":


    ####################
    # Load data, do PCA

    sdfs   = np.load("sdfs.npy").reshape(500,-1)
    r2s    = np.load("r2s.npy")
    labels = np.load("tr_te.npy")
    colors = np.array(["blue","red"])[labels]
    tr_idx  = labels == 0
    te_idx  = labels == 1

    n_pc = 50
    pca = PCA(n_components=n_pc)
    pca.fit(sdfs[tr_idx,:]);
    scores = pca.score_samples(sdfs)
    Xt = pca.transform(sdfs)

    plt.figure(dpi=200, figsize=(6,3))

    plt.bar(1+np.arange(len(pca.explained_variance_ratio_)),100*pca.explained_variance_ratio_)
    plt.xlabel("SDF Principal Components")
    plt.ylabel("% of Explained Variance")

    plt.savefig("sdf-pca.png", bbox_inches="tight")
    plt.close()

    #
    ####################


    ####################
    # OOD detection

    sets = dict(tr=scores[tr_idx], te=scores[te_idx])
    curve_colors = dict(tr="blue", te="red")
    labels = dict(tr=f'Training, median: {np.median(scores[tr_idx]):.2f}', te=f'Testing, median: {np.median(scores[te_idx]):.2f}')
    plt.figure(dpi=200,figsize=(5,3))
    for key in sets:
        bins = 30
        plt.hist(sets[key], bins=bins, density=True, histtype="step", linewidth=2., edgecolor=curve_colors[key], label=labels[key])
    plt.legend(loc="upper left")
    plt.xlabel("Log-likelihood score")
    plt.ylabel("Probability Density")

    plt.savefig("loglikelihood-hist.png", bbox_inches="tight")
    plt.close()

    threshold = np.percentile(scores[tr_idx],5) # 5th percentile of log-likelihood on training data
    print(f"Log-Likelihood OOD Threshold = {threshold}")

    #
    ####################


    ####################
    # R2 Histogram

    sets = dict(tr=r2s[tr_idx], te=r2s[te_idx])
    colors = dict(tr="blue", te="red")
    labels = dict(tr=f'Training, median: {np.median(r2s[tr_idx]):.2f}', te=f'Testing, median: {np.median(r2s[te_idx]):.2f}')
    plt.figure(dpi=200,figsize=(5,3))
    for key in sets:
        r2s = sets[key]
        bins = 30
        plt.hist(r2s, bins=bins, density=True, histtype="step", linewidth=2., edgecolor=colors[key], label=labels[key])
    plt.legend(loc="upper left")
    plt.xlabel("$R^2$")
    plt.ylabel("Probability Density")

    plt.savefig("r2-hist.png", bbox_inches="tight")
    plt.close()

    #
    ####################


    ####################
    # R2 Prediction with OOD

    r2_model = KNeighborsRegressor(n_neighbors=3)

    X_train = Xt[tr_idx]
    y_train = r2s[tr_idx]
    X_test = Xt[te_idx]
    y_test = r2s[te_idx]

    r2_model.fit(X_train, y_train)
    y_pred = r2_model.predict(X_test)
    ytr_pred = r2_model.predict(X_train)
    pred = r2_model.predict(Xt)

    plt.figure(dpi=300)

    plt.scatter(
        y_train, ytr_pred,
        color='blue', edgecolors='black', linewidths=0.3,
        s=7, label='Train predictions'
    )

    plt.scatter(
        y_test, y_pred,
        color='red', edgecolors='black', linewidths=0.3,
        s=7, label='Test predictions'
    )

    highlight_mask = scores < threshold
    plt.scatter(
        r2s[highlight_mask], pred[highlight_mask],
        facecolors='none', edgecolors='orange', linewidths=1.0,
        s=30, label='OOD'
    ) # Circle OOD points

    min_val = .6
    max_val = 1
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label='Predicted = Actual', zorder=-1)

    plt.xlabel("Actual $R^2$")
    plt.ylabel("Predicted $R^2$")

    plt.legend(loc='lower left')
    plt.savefig("r2-ood.png",bbox_inches="tight")
    plt.close()

    #
    ####################
