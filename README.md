![Graphical Abstract](https://ars.els-cdn.com/content/image/1-s2.0-S0957417425019463-ga1_lrg.jpg)

---

# 🧠 BiGCN-CLIP Fusion Net (BCFN) for Rumor Detection

This repository contains a minimal implementation of the BCFN model proposed in our paper:

"Multimodal Fusion for Rumor Sleuthing: A Comprehensive Approach"  
*Mohammad-Reza Farahi, Fateme Jafarinejad*  
Published in *Expert Systems with Applications, 2025*

📄 [Paper Link](https://doi.org/10.1016/j.eswa.2025.128327)  
📧 Correspondence: jafarinejad@shahroodut.ac.ir

---

## 🧰 About the Project

The BCFN (BiGCN-CLIP Fusion Net) model fuses graph structure and text semantics for robust rumor detection on social media. It integrates:

- 🔁 Bi-Directional GCNs (Bi-GCN) for modeling rumor propagation networks
- ✍️ CLIP (text encoder) for extracting rich semantic features from textual posts
- 🎯 Cross-modal multi-head attention for fusing modalities
- 🧮 A lightweight MLP classifier for final prediction

---

## 📊 Datasets

We evaluate BCFN on three benchmark rumor detection datasets:

- Twitter15
- Twitter16
- Weibo

> ⚠️ Due to licensing issues, raw datasets are not included. Please follow the original papers to obtain them.

---

## 📦 Requirements

This code was tested using:

- Python ≥ 3.9  
- PyTorch ≥ 1.12  
- PyTorch Geometric (PyG)  
- Transformers (for CLIP)  
- scikit-learn  
- tqdm


---

🧪 Results

Our model achieves state-of-the-art performance on all datasets. See the paper for full tables and ablation studies.


---

📌 Citation

If you use this code or paper in your work, please cite:

```
@article{farahi2025bcfn,
  title={Multimodal fusion for rumor sleuthing: A comprehensive approach},
  author={Farahi, Mohammad-Reza and Jafarinejad, Fateme},
  journal={Expert Systems with Applications},
  volume={288},
  year={2025},
  doi={10.1016/j.eswa.2025.128327}
}
```

---

📚 Acknowledgements

CLIP by OpenAI

Bi-GCN implementation inspired by [Bian et al., 2020]

Datasets from [Ma et al., 2016, 2017]



---

🛠 Status

This is a quick and dirty release of the source code.
Bug reports, feature requests, or contributions are welcome!


---

☕ Contact

For questions, feel free to reach out via email:
📧 rqlzienc@gmail.com
