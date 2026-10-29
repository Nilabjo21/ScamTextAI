"""Generate the evidence-based CA-3 report and an editable architecture diagram."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
REPORT_PATH = ROOT / "ScamText_AI_CA3_Report_v2.docx"


RECENT_STUDIES = [
    (
        "(Oswald et al., 2022)",
        "SMS spam; UCI SMS Spam Collection. Intention labels, contextual embeddings from pretrained NLP models, and supervised classifiers.",
        "98.07% accuracy; precision and recall approximately 0.97. The abstract describes stable performance but says accuracy was not the best among compared approaches.",
        "No explicit limitation in the accessible abstract.",
    ),
    (
        "(Bountakas & Xenakis, 2022)",
        "Phishing email; 32,051 benign and 3,460 phishing samples. Hybrid email-content/trait features with stacking and soft-voting ensembles.",
        "F1 = 0.9942, as reported in the abstract.",
        "Dataset name and study limitations were not established in the accessible record.",
    ),
    (
        "(Stevanović, 2022)",
        "Phishing email; character and word embeddings are identified by the article title.",
        "Dataset, detailed method, and metrics were not verifiable from an accessible abstract in this review.",
        "Could not verify study limitations without accessible full text.",
    ),
    (
        "(Ghatasheh et al., 2022)",
        "Spam prediction; Twitter spam is the primary setting and the method is also applied to SMS. Modified genetic algorithm for XGBoost feature selection and hyperparameter optimization.",
        "For the main spam prediction task, average geometric mean 82.32% and accuracy 92.67%; the abstract gives no SMS-specific metric.",
        "The abstract discusses class imbalance and feature-space size as challenges, but no specific limitation of the study is stated.",
    ),
    (
        "(Atawneh & Aljehani, 2023)",
        "Phishing/benign email. CNN, LSTM, RNN, and BERT models using NLP-derived features.",
        "The best reported model (BERT with LSTM) achieved 99.61% accuracy.",
        "Dataset identity and study limitations were not reported in the accessible abstract.",
    ),
    (
        "(Thapa et al., 2023)",
        "Phishing email; datasets not named in the accessible abstract. Federated RNN/BERT experiments compare organization count and balanced versus asymmetric distributions.",
        "With fixed total data, RNN accuracy fell 1.8% from 2 to 10 organizations; BERT accuracy rose 0.6% from 2 to 5 organizations.",
        "Highly asymmetric email distributions produced unstable outputs and degraded global-model performance.",
    ),
    (
        "(Qi et al., 2023)",
        "Phishing email; dataset not named in accessible abstract. FMPED/FMMPED ensembles with undersampling of benign emails in overlapping regions.",
        "F1 = 0.9945, accuracy = 0.9945, AUC = 0.9828, and G-mean = 0.9827.",
        "No explicit limitation was reported in the accessible abstract.",
    ),
    (
        "(Milner & Baron, 2023)",
        "Text-message phishing on UCI SMS Spam Collection (accessed via Kaggle), with a Twitter spam set for evaluation. RoBERTa-base and sentence-transformer/scikit-learn methods.",
        "Best SMS F1 = 0.938; the accessible abstract gives no Twitter value.",
        "No explicit study limitation was stated in the accessible abstract.",
    ),
    (
        "(Altwaijry et al., 2024)",
        "Phishing Corpus and SpamAssassin email datasets. 1D-CNN combined with LSTM, Bi-LSTM, GRU, or Bi-GRU.",
        "The reported advanced 1D-CNN/Bi-GRU model achieved 99.68% accuracy, 99.66% F1, 99.32% recall, and 100% precision. The abstract notes performance decline with additional depth after an initial improvement.",
        "No explicit study limitation was stated in the accessible abstract.",
    ),
    (
        "(Mambina et al., 2024)",
        "Swahili SMS from Tanzanian telecommunications companies and the English UCI SMS corpus. Deep-learning models with 10-fold cross-validation.",
        "The abstract reports accuracy 99.98 for CNN-LSTM-LSTM on Swahili data and 98.38 for CNN-BiLSTM on UCI.",
        "No explicit limitation was stated in the accessible abstract; broader generalization was not established.",
    ),
    (
        "(Latifah et al., 2024)",
        "Indonesian SMS spam with six categories; dataset name and size not given in the accessible abstract. IndoBERT with Easy Data Augmentation.",
        "IndoBERT accuracy 91%; augmentation reportedly raised minority-class F1 by an average 12% and overall accuracy to 93%.",
        "No explicit study limitation was stated in the accessible abstract.",
    ),
    (
        "(Al-Subaiey et al., 2024)",
        "Email spam classification in a web platform with explainable AI; the accessible abstract does not identify the dataset.",
        "F1 = 0.99, as reported in the accessible abstract.",
        "Dataset details and explicit study limitations were not reported in the accessible abstract.",
    ),
]


REFERENCES = [
    "Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering. In Proceedings of the 11th ACM Symposium on Document Engineering (pp. 259–262). Association for Computing Machinery. https://doi.org/10.1145/2034691.2034742",
    "Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Uz Zaman, S. M. A. (2024). Novel interpretable and robust web-based AI platform for phishing email detection. Computers & Electrical Engineering, 120, Article 109625. https://doi.org/10.1016/j.compeleceng.2024.109625",
    "Altwaijry, N., Al-Turaiki, I. M., Alotaibi, R., & Alakeel, F. (2024). Advancing phishing email detection: A comparative study of deep learning models. Sensors, 24(7), Article 2077. https://doi.org/10.3390/s24072077",
    "Atawneh, S., & Aljehani, H. (2023). Phishing email detection model using deep learning. Electronics, 12(20), Article 4261. https://doi.org/10.3390/electronics12204261",
    "Bountakas, P., & Xenakis, C. (2022). HELPHED: Hybrid ensemble learning phishing email detection. Journal of Network and Computer Applications, 210, Article 103545. https://doi.org/10.1016/j.jnca.2022.103545",
    "Cybersecurity and Infrastructure Security Agency. (n.d.). Recognize and report phishing. https://www.cisa.gov/secure-our-world/recognize-and-report-phishing",
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423",
    "Federal Bureau of Investigation, Internet Crime Complaint Center. (2025). 2024 IC3 annual report. https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf",
    "Ghatasheh, N., Al-Taharwa, I. A., & Aldebei, K. (2022). Modified genetic algorithm for feature selection and hyper parameter optimization: Case of XGBoost in spam prediction. IEEE Access, 10, 84365–84383. https://doi.org/10.1109/ACCESS.2022.3196905",
    "Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804",
    "Latifah, N., Dwiyansaputra, R., & Nugraha, G. S. (2024). Multiclass text classification of Indonesian short message service (SMS) spam using deep learning method and easy data augmentation. Matrik: Jurnal Manajemen, Teknik Informatika dan Rekayasa Komputer, 23(3), 663–676. https://doi.org/10.30812/matrik.v23i3.3835",
    "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (Vol. 30). https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
    "Mambina, I. S., Ndibwile, J. D., Uwimpuhwe, D., & Michael, K. F. (2024). Uncovering SMS spam in Swahili text using deep learning approaches. IEEE Access, 12, 25164–25175. https://doi.org/10.1109/ACCESS.2024.3365193",
    "Milner, H., & Baron, M. (2023). Establishing an optimal online phishing detection method: Evaluating topological NLP transformers on text message data. Journal of Data Science and Intelligent Systems, 2(1), 37–45. https://doi.org/10.47852/bonviewjdsis32021131",
    "Oswald, C., Simon, S. E., & Bhattacharya, A. (2022). SpotSpam: Intention analysis-driven SMS spam detection using BERT embeddings. ACM Transactions on the Web, 16(3). https://doi.org/10.1145/3538491",
    "Qi, Q., Wang, Z., Xu, Y., Fang, Y., & Wang, C. (2023). Enhancing phishing email detection through ensemble learning and undersampling. Applied Sciences, 13(15), Article 8756. https://doi.org/10.3390/app13158756",
    "Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?”: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135–1144). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939778",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432",
    "Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv. https://doi.org/10.48550/arXiv.1910.01108",
    "scikit-learn developers. (n.d.). TfidfVectorizer. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html",
    "Stevanović, N. (2022). Character and word embeddings for phishing email detection. Computing and Informatics, 41(5), 1337–1357. https://doi.org/10.31577/cai_2022_5_1337",
    "Tabassi, E. (2023). Artificial intelligence risk management framework (AI RMF 1.0) (NIST AI 100-1). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.AI.100-1",
    "Thapa, C., Tang, J. W., Abuadbba, A., Gao, Y., Camtepe, S., Nepal, S., Almashor, M., & Zheng, Y. (2023). Evaluation of federated learning in phishing email detection. Sensors, 23(9), Article 4346. https://doi.org/10.3390/s23094346",
    "UCI Machine Learning Repository. (2011). SMS Spam Collection [Data set]. https://doi.org/10.24432/C5CC84",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (Vol. 30). https://papers.nips.cc/paper/7181-attention-is-all-you-need",
    "Yeafi, A. (2023). Spam email classification [Data set]. Kaggle. https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification",
]


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def create_architecture_figure(path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    nodes = [
        (0.25, 3.4, 2.0, 1.05, "Verified data\nUCI SMS + optional CSVs", "#dae8fc"),
        (2.65, 3.4, 2.0, 1.05, "Validate labels\nDeduplicate / split", "#d5e8d4"),
        (5.05, 3.4, 2.0, 1.05, "Normalize text\nTrain-only TF-IDF", "#fff2cc"),
        (7.45, 3.4, 2.0, 1.05, "Train LR + NB\nTune on validation", "#e1d5e7"),
        (9.85, 3.4, 1.9, 1.05, "Test metrics\nSave artifacts", "#f8cecc"),
        (0.25, 0.65, 2.0, 1.05, "User text\nor CSV upload", "#dae8fc"),
        (2.65, 0.65, 2.0, 1.05, "Same normalized\nfitted TF-IDF", "#fff2cc"),
        (5.05, 0.65, 2.0, 1.05, "Saved Logistic\nRegression", "#e1d5e7"),
        (7.45, 0.65, 2.0, 1.05, "Prediction +\nexplainable signals", "#fff2cc"),
        (9.85, 0.65, 1.9, 1.05, "Streamlit\ninterface", "#d5e8d4"),
    ]
    for x, y, width, height, label, color in nodes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#52677a",
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    for start, end, y in [
        (2.25, 2.65, 3.925), (4.65, 5.05, 3.925),
        (7.05, 7.45, 3.925), (9.45, 9.85, 3.925),
        (2.25, 2.65, 1.175), (4.65, 5.05, 1.175),
        (7.05, 7.45, 1.175), (9.45, 9.85, 1.175),
    ]:
        ax.add_patch(FancyArrowPatch((start, y), (end, y), arrowstyle="-|>", mutation_scale=12, color="#52677a"))
    ax.add_patch(FancyArrowPatch((10.8, 3.4), (6.0, 1.78), connectionstyle="arc3,rad=-0.18",
                                 arrowstyle="-|>", mutation_scale=12, color="#8a6d3b", linestyle="--"))
    ax.text(8.55, 2.55, "saved model + vectorizer", ha="center", fontsize=8, color="#6b5a34")
    ax.text(6, 4.85, "ScamText AI: Training and Inference Architecture", ha="center", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_document_styles(document):
    section = document.sections[0]
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for style_name, size in [("Title", 22), ("Heading 1", 15), ("Heading 2", 12)]:
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True


def add_table(document, headers, rows, font_size=8.5):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(font_size)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(font_size)
    return table


def add_caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)


def make_report(include_application_screenshot=False):
    stats = load_json(OUTPUTS / "dataset_statistics.json")
    split_stats = load_json(OUTPUTS / "split_statistics.json")
    results = load_json(OUTPUTS / "results.json")
    comparison = pd.read_csv(OUTPUTS / "model_comparison.csv", index_col=0)
    data = pd.read_csv(ROOT / "data" / "combined_dataset.csv")
    raw_sources = data.groupby(["source", "label"]).size().unstack(fill_value=0)
    raw_sources = raw_sources.rename(columns={0: "Legitimate", 1: "Spam/scam"})
    input_sources = stats["input_counts_by_source_before_deduplication"]
    uci_counts = input_sources.get(
        "UCI_SMS_Spam_Collection",
        {"legitimate": 0, "spam_or_scam": 0},
    )
    kaggle_sources = {
        source: counts
        for source, counts in input_sources.items()
        if source != "UCI_SMS_Spam_Collection"
    }
    source_description = (
        f"The UCI archive yielded {uci_counts['legitimate'] + uci_counts['spam_or_scam']:,} "
        f"parsed labels ({uci_counts['legitimate']:,} ham and "
        f"{uci_counts['spam_or_scam']:,} spam)."
    )
    for source, counts in kaggle_sources.items():
        duplicates = stats["duplicate_counts_by_source"][source]
        source_description += (
            f" The user-supplied Kaggle CSV '{source}' (download date not recorded) added "
            f"{counts['legitimate'] + counts['spam_or_scam']:,} explicitly labeled rows "
            f"({counts['legitimate']:,} ham and {counts['spam_or_scam']:,} spam); "
            f"{duplicates['repeated_from_prior_sources']:,} unique records duplicated "
            "an earlier source."
        )
    if kaggle_sources:
        source_description += (
            " Kaggle lists Apache 2.0 for this source, but substantial message overlap "
            "means its underlying provenance and redistribution rights are not "
            "independently established here (Yeafi, 2023)."
        )
        license_description = (
            "The UCI record describes corpus sources as free or free for research, but "
            "does not expose one uniform license for all component sources. The Kaggle "
            "listing declares Apache 2.0; however, the observed overlap with UCI means "
            "underlying provenance and redistribution rights are not independently "
            "established here (Yeafi, 2023). Neither original source corpus is "
            "redistributed by this report."
        )
    else:
        source_description += " No Kaggle CSV was included in this run."
        license_description = (
            "The UCI record describes corpus sources as free or free for research, but "
            "does not expose one uniform license for all component sources. The source "
            "data is therefore not redistributed by this report."
        )

    create_architecture_figure(OUTPUTS / "system_architecture.png")
    document = Document()
    set_document_styles(document)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("\n\nSCAMTEXT AI\n").bold = True
    title.add_run(
        "An NLP-Based Framework for SMS Spam Detection and "
        "Explainable Message Screening\n\n"
    ).font.size = Pt(15)
    title.add_run("Natural Language Processing Lab (TEE7100)\n")
    title.add_run("CA-3 Project Report\n\n")
    title.add_run("Student Name: Nilabjo Goswami | PRN: 23070122149\n")
    title.add_run("Student Name: Joshua Bara | PRN: 23070122112\n")
    title.add_run("Group Number: 65\n\n")
    title.add_run("Department of Computer Science and Engineering\n")
    title.add_run("Symbiosis Institute of Technology, Pune\n")
    title.add_run("Academic Year 2026–27\n")
    title.add_run("Submission date: 13 October 2026")
    document.add_page_break()

    abstract = (
        f"Online scam and spam messages use deceptive language, urgency, and requests "
        f"for action to influence recipients. This project implements ScamText AI, an "
        f"explainable binary text-classification prototype that flags spam-like messages "
        f"and displays lexical warning signals. {source_description} After exact "
        f"deduplication and text normalization, {split_stats['total_samples']:,} "
        f"messages were split into training, validation, and held-out test sets. URL, "
        f"currency, and phone-number signals are retained as tokens. TF-IDF unigram/bigram "
        f"features feed Logistic Regression and Multinomial Naive Bayes; thresholds are "
        f"selected on validation data by F1, and final metrics are computed on test data. "
        f"Logistic Regression obtained accuracy {results['Logistic Regression + TF-IDF']['metrics']['accuracy']:.3f}, "
        f"F1 {results['Logistic Regression + TF-IDF']['metrics']['f1']:.3f}, recall "
        f"{results['Logistic Regression + TF-IDF']['metrics']['recall']:.3f}, and ROC-AUC "
        f"{results['Logistic Regression + TF-IDF']['metrics']['roc_auc']:.3f}; Naive Bayes "
        f"obtained F1 {results['Naive Bayes + TF-IDF']['metrics']['f1']:.3f}. The Streamlit "
        f"interface uses the trained Logistic Regression model and presents model evidence "
        f"separately from rule-based indicators. No transformer or scam-category model was "
        f"trained because the available labels are binary. The experiment addresses class "
        f"imbalance through validation threshold selection and reports precision, recall, "
        f"F1, and ROC-AUC alongside accuracy. Model scores are not calibrated guarantees "
        f"and warning terms do not establish that a message is fraudulent. Future work "
        f"should use independently sourced data, temporal testing, score calibration, and "
        f"multilingual evaluation. Results remain specific to these corpora and this split."
    )
    document.add_heading("Abstract", level=1)
    document.add_paragraph(abstract)
    document.add_paragraph(
        "Keywords: SMS spam detection, text classification, TF-IDF, explainable NLP, "
        "Logistic Regression, Naive Bayes"
    )

    document.add_heading("1. Introduction", level=1)
    document.add_paragraph(
        "Short messages and emails can impersonate trusted organizations, offer false "
        "rewards, or pressure recipients into clicking links and disclosing information. "
        "Such social-engineering attempts exploit language and timing rather than only "
        "technical vulnerabilities. Public guidance emphasizes recognizing suspicious "
        "requests and reporting phishing, while cybercrime reporting provides a broader "
        "operational context for the continuing threat (Cybersecurity and Infrastructure "
        "Security Agency, n.d.; Federal Bureau of Investigation, Internet Crime Complaint "
        "Center, 2025)."
    )
    document.add_paragraph(
        "Automated message screening can help individuals, students, employees, service "
        "providers, and online platforms prioritize suspicious content. Yet messages are "
        "short, informal, and highly variable. Legitimate alerts can be urgent, and "
        "ordinary words such as “account” or “claim” are not proof of fraud. A classifier "
        "therefore needs measurable evaluation and should expose warning evidence without "
        "presenting a score as a guarantee."
    )
    document.add_paragraph(
        "Prior work investigates SMS spam, phishing email, and message classification "
        "using traditional features, contextual embeddings, deep networks, and ensembles "
        "(Oswald et al., 2022; Bountakas & Xenakis, 2022; Thapa et al., 2023; "
        "Altwaijry et al., 2024). Reported high scores occur on different datasets and "
        "validation protocols, so they are not directly comparable. The literature also "
        "covers multilingual SMS and separate email settings; a result on one corpus does "
        "not by itself demonstrate transfer to another (Mambina et al., 2024; Latifah et "
        "al., 2024)."
    )
    document.add_paragraph(
        "This project addresses a practical reproducibility gap: demonstrate an end-to-end "
        "binary message-screening prototype whose source data, deduplication, train/ "
        "validation/test split, threshold selection, and held-out results are explicit. "
        "It compares two TF-IDF baselines using UCI SMS data and the user's labeled "
        "Kaggle CSV, with substantial cross-source message overlap measured and "
        "reported. The Streamlit interface provides lexical red flags and model feature "
        "contributions. Its scope is deliberately limited to the labels and data "
        "actually available."
    )
    document.add_paragraph("Major contributions:")
    for item in [
        "A source-aware data loader that validates binary labels and rejects synthetic or guessed training records.",
        "A stratified, leakage-conscious TF-IDF workflow with validation-only threshold selection and held-out evaluation.",
        "A comparison of Logistic Regression and Multinomial Naive Bayes on the same test split.",
        "A working single-message/batch Streamlit interface with separate model evidence and lexical warning indicators.",
    ]:
        document.add_paragraph(item, style="List Bullet")

    document.add_heading("2. Literature Review", level=1)
    document.add_paragraph(
        "Recent studies cover SMS spam and phishing emails using contextual language "
        "representations, recurrent/convolutional networks, ensemble learning, "
        "undersampling, and federated evaluation. The studies below report their own "
        "datasets and metrics as available from accessible abstracts or publisher "
        "records. Where details could not be verified, this report explicitly marks them "
        "as unreported rather than inferring them."
    )
    literature_rows = [
        (paper, setting, evidence, limitations)
        for paper, setting, evidence, limitations in RECENT_STUDIES
    ]
    add_table(
        document,
        ["Study", "Problem, data, and method", "Reported evidence", "Limitations / evidence gaps"],
        literature_rows,
        font_size=7.2,
    )
    document.add_paragraph(
        "Synthesis and research gap. The reviewed work demonstrates that multiple model "
        "families can perform well on their chosen corpora, but the reported numbers use "
        "different datasets, label definitions, class balances, and validation protocols. "
        "For example, SMS embedding work, phishing-email ensembles, and Swahili/Indonesian "
        "SMS experiments address distinct settings; their headline metrics cannot be "
        "ranked as if they arose from one benchmark (Oswald et al., 2022; Qi et al., 2023; "
        "Mambina et al., 2024; Latifah et al., 2024). The accessible abstracts often omit "
        "dataset identity or explicit limitations, which further restricts comparison."
    )
    document.add_paragraph(
        "Accordingly, this project does not claim a new classifier or a new scam taxonomy. "
        "Its bounded contribution is a reproducible baseline comparison and a user-facing "
        "prototype that keeps binary corpus labels, model outputs, and lexical indicators "
        "distinct. This choice avoids deriving unsupported scam categories from datasets "
        "that only label messages as spam or ham. It also follows reproducibility "
        "recommendations to document splits and guard against leakage (Kapoor & "
        "Narayanan, 2023)."
    )
    document.add_paragraph(
        "Foundational and methodological sources complement the recent studies: the SMS "
        "corpus and prior SMS filtering work establish the dataset context (UCI Machine "
        "Learning Repository, 2011; Almeida et al., 2011); BERT/Transformers underpin "
        "contextual alternatives (Devlin et al., 2019; Vaswani et al., 2017), while "
        "DistilBERT is an optional future experiment rather than a result of this run "
        "(Sanh et al., 2019). LIME and SHAP are relevant explanation frameworks but are "
        "not claimed as implemented in this prototype (Ribeiro et al., 2016; Lundberg & "
        "Lee, 2017)."
    )

    document.add_heading("3. Proposed Methodology", level=1)
    document.add_heading("3.1 System architecture", level=2)
    document.add_paragraph(
        "Figure 1 shows the implemented training and inference paths. The editable "
        "diagrams.net source is supplied as outputs/system_architecture.drawio."
    )
    document.add_picture(str(OUTPUTS / "system_architecture.png"), width=Inches(6.6))
    add_caption(document, "Figure 1. ScamText AI data preparation, model training, evaluation, and inference architecture.")
    document.add_heading("3.2 Dataset and label definition", level=2)
    document.add_paragraph(
        "The experiment uses the SMS Spam Collection obtained from the UCI Machine "
        "Learning Repository (UCI Machine Learning Repository, 2011). The archive "
        "contains 5,574 newline-separated lines; the tabular parser produced "
        f"{uci_counts['legitimate'] + uci_counts['spam_or_scam']:,} UCI labels. "
        f"{source_description} Across the loaded data, exact deduplication removed "
        f"{stats['duplicates_removed']:,} repeated messages, leaving {stats['samples']:,} "
        f"unique labeled records. Text normalization and a second duplicate check "
        f"produced {split_stats['total_samples']:,} modeling examples. The task is binary "
        f"spam versus ham; no reliable scam-category labels are supplied. {license_description}"
    )
    skipped_rows = stats.get("skipped_source_rows", {})
    if skipped_rows:
        skipped_count = sum(source["rows"] for source in skipped_rows.values())
        document.add_paragraph(
            f"Data-quality handling excluded {skipped_count} row(s) with unsupported "
            f"non-binary labels or empty text. Exact counts by file are recorded in "
            "outputs/dataset_statistics.json; labels were not inferred."
        )
    source_rows = []
    for source, row in raw_sources.iterrows():
        raw_counts = input_sources.get(source, {"legitimate": 0, "spam_or_scam": 0})
        duplicates = stats["duplicate_counts_by_source"].get(
            source,
            {"repeated_from_prior_sources": 0},
        )
        source_rows.append((
            source,
            raw_counts["legitimate"] + raw_counts["spam_or_scam"],
            f"{int(row.get('Legitimate', 0)):,} / "
            f"{int(row.get('Spam/scam', 0)):,} / {int(row.sum()):,}",
            f"{duplicates['repeated_from_prior_sources']:,} prior / "
            f"{duplicates['repeated_within_source']:,} within",
        ))
    add_table(
        document,
        ["Source", "Valid labeled rows", "Unique ham / spam / total", "Duplicates removed: prior / within"],
        source_rows,
    )
    document.add_heading("3.3 Preprocessing and feature engineering", level=2)
    document.add_paragraph(
        "Messages are lowercased and whitespace-normalized. Email addresses, URLs, "
        "phone-number patterns, and currency symbols are represented with stable tokens "
        "rather than deleted, preserving structural evidence that may aid classification. "
        "No NLTK downloads, stemming, lemmatization, or stop-word removal are required. "
        "TF-IDF with word unigrams/bigrams, sublinear term frequency, a 10,000-feature "
        "cap, max_df = 0.98, and min_df = 2 is fitted on the training split only "
        "(scikit-learn developers, n.d.)."
    )
    document.add_heading("3.4 Model development and explainability", level=2)
    document.add_paragraph(
        "Logistic Regression (maximum 1,000 iterations, random_state 42) and "
        "Multinomial Naive Bayes are trained on the same sparse TF-IDF training matrix. "
        "The validation split selects a separate decision threshold for each model by "
        "maximizing F1. The test split is reserved for the final comparison. The "
        "application presents the Logistic Regression class score and the strongest "
        "weighted TF-IDF coefficients for the predicted class. Lexical indicators "
        "(urgency, prize terms, financial terms, suspicious links, currency, phone, and "
        "personal-information requests) are a separate heuristic explanation layer; "
        "they do not override the learned classification. No LIME, SHAP, supervised "
        "multiclass classification, or transformer results are claimed."
    )

    document.add_heading("4. Experimental Setup", level=1)
    document.add_paragraph(
        f"Execution environment: {platform.platform()}; Python {platform.python_version()}; "
        "pandas, NumPy, scikit-learn, joblib, Matplotlib, Seaborn, and Streamlit. Baseline "
        "training and inference ran on CPU; no GPU or transformer training was used. "
        "The split is stratified and uses random_state = 42: "
        f"{split_stats['train_samples']:,} train, {split_stats['validation_samples']:,} "
        f"validation, and {split_stats['test_samples']:,} test examples. The validation "
        "threshold is chosen by F1, then held-out test accuracy, precision, recall, F1, "
        "ROC-AUC, and confusion matrices are computed. Accuracy is read with caution "
        "under class imbalance; precision/recall and confusion matrices are also reported "
        "(Saito & Rehmsmeier, 2015). Evaluation and deployment claims should be "
        "interpreted under responsible-AI and risk-management guidance (Tabassi, 2023)."
    )

    document.add_heading("5. Results and Discussion", level=1)
    document.add_paragraph(
        f"The final test set contains {split_stats['test_samples']:,} messages. Logistic "
        f"Regression provides the higher recall and F1 of the two measured baselines. "
        f"Naive Bayes produces higher precision but misses more positive examples at its "
        f"validation-selected threshold. Both models achieve high ROC-AUC on this random "
        f"held-out split; this does not establish temporal or independent cross-domain "
        f"generalization."
    )
    metric_rows = []
    for name, row in comparison.iterrows():
        metric_rows.append((
            name,
            f"{row['accuracy']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1']:.4f}",
            f"{row['roc_auc']:.4f}",
            f"{row['threshold']:.4f}",
        ))
    add_table(
        document,
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Validation threshold"],
        metric_rows,
    )
    figures = [
        ("confusion_matrices.png", "Figure 2. Confusion matrices for Logistic Regression and Naive Bayes on the held-out test split."),
        ("roc_curves.png", "Figure 3. ROC curves computed from held-out test scores."),
    ]
    if include_application_screenshot:
        figures.append((
            "application_screenshot.png",
            "Figure 4. Streamlit single-message result for the displayed example; output is generated by the trained classifier.",
        ))
    for filename, caption in figures:
        figure_path = OUTPUTS / filename
        if figure_path.is_file():
            document.add_picture(str(figure_path), width=Inches(6.3))
            add_caption(document, caption)
    logistic_matrix = results["Logistic Regression + TF-IDF"]["confusion_matrix"]
    naive_bayes_matrix = results["Naive Bayes + TF-IDF"]["confusion_matrix"]
    document.add_paragraph(
        f"For Logistic Regression, the confusion matrix contains "
        f"{logistic_matrix[0][0]} true negatives, {logistic_matrix[0][1]} false "
        f"positives, {logistic_matrix[1][0]} false negatives, and "
        f"{logistic_matrix[1][1]} true positives. Naive Bayes contains "
        f"{naive_bayes_matrix[0][0]} true negatives, {naive_bayes_matrix[0][1]} "
        f"false positives, {naive_bayes_matrix[1][0]} false negatives, and "
        f"{naive_bayes_matrix[1][1]} true positives. These counts show the "
        f"precision/recall trade-off under the chosen validation thresholds. They "
        f"describe this split only, not production error rates."
    )

    document.add_heading("6. Comparative Analysis", level=1)
    document.add_paragraph(
        "The two baselines use identical data partitions and the same TF-IDF representation, "
        "so the comparison isolates classifier and threshold differences more cleanly "
        "than comparisons across unrelated corpora. Logistic Regression reaches F1 "
        f"{results['Logistic Regression + TF-IDF']['metrics']['f1']:.3f} and recall "
        f"{results['Logistic Regression + TF-IDF']['metrics']['recall']:.3f}; Naive Bayes "
        f"reaches F1 {results['Naive Bayes + TF-IDF']['metrics']['f1']:.3f} and precision "
        f"{results['Naive Bayes + TF-IDF']['metrics']['precision']:.3f}. A threshold "
        "selected on validation data was important: the default 0.5 threshold had lower "
        "positive-class recall during the initial diagnostic run. The revised test "
        "scores use the validation-selected threshold, not a threshold selected from the "
        "test set."
    )
    document.add_paragraph(
        "Compared with the recent neural and ensemble studies in Section 2, these "
        "baselines are lighter to train and easier to inspect, but this experiment does "
        "not establish that they outperform those papers. The datasets and protocols are "
        "not aligned. A fair transformer comparison would require a controlled run with "
        "the same source records and split; DistilBERT remains optional and unmeasured "
        "here. Explainability is limited to lexical signals and linear feature "
        "contributions, rather than post-hoc LIME/SHAP analysis."
    )

    document.add_heading("9. Limitations", level=1)
    for item in [
        "The primary source is an English SMS spam corpus. The additional CSV is highly overlapping and does not provide independently validated email-domain evidence; the combined data is not a dedicated, current online-scam benchmark.",
        "The dataset is class-imbalanced and contains repeated messages. Exact deduplication was applied, but near-duplicate and source-related leakage may remain.",
        "A stratified random split evaluates in-corpus generalization, not performance on future scam tactics, a separate dataset, or a different language.",
        "Binary spam/ham labels do not support trained prediction of phishing, lottery, job, investment, or other scam categories.",
        "The word-based indicator layer is heuristic and may match benign messages; the model score is not a calibrated guarantee or a URL safety verdict.",
        "No transformer experiment, temporal evaluation, or human review study was completed in the measured run.",
    ]:
        document.add_paragraph(item, style="List Bullet")

    document.add_heading("10. Future Scope", level=1)
    document.add_paragraph(
        "Future work should first add verified, independently sourced email and SMS "
        "corpora with documented licenses and label definitions, then report each source "
        "separately as well as in a combined experiment. Temporal and cross-dataset "
        "testing would measure whether the model generalizes beyond the random split. "
        "A controlled DistilBERT experiment, probability calibration, precision-recall "
        "analysis, and human-reviewed scam-category labels could extend the current "
        "baseline. Multilingual and code-mixed Hindi-English data, privacy-aware "
        "deployment, and regular model updates are additional directions. Any deployment "
        "should preserve user review and clear uncertainty communication rather than "
        "treating a score as an authoritative verdict (Tabassi, 2023)."
    )

    document.add_heading("11. Conclusion", level=1)
    conclusion = (
        f"This project implemented ScamText AI as a reproducible binary SMS spam-screening "
        f"pipeline and a working Streamlit prototype. The final modeling dataset contained "
        f"{split_stats['total_samples']:,} messages after exact deduplication and text "
        f"normalization, with {stats['classes']['legitimate']:,} legitimate examples and "
        f"{stats['classes']['spam_or_scam']:,} spam-like examples. The raw source totals "
        f"({stats['input_samples_before_deduplication']:,} labeled rows before deduplication) "
        f"were not used as the modeling set and are reported only as raw input counts. "
        f"The final split was {split_stats['train_samples']:,} training, "
        f"{split_stats['validation_samples']:,} validation, and {split_stats['test_samples']:,} "
        f"test messages, supporting a stratified held-out evaluation.\n\n"
        f"Text cleaning preserved tokens such as URLs, phone numbers, and currency markers, and "
        f"TF-IDF unigram/bigram features were fit on training data only. Logistic Regression "
        f"and Multinomial Naive Bayes were trained and evaluated on the same test split, with "
        f"validation-selected thresholds based on F1. Logistic Regression achieved accuracy "
        f"{results['Logistic Regression + TF-IDF']['metrics']['accuracy']:.3f}, precision "
        f"{results['Logistic Regression + TF-IDF']['metrics']['precision']:.3f}, recall "
        f"{results['Logistic Regression + TF-IDF']['metrics']['recall']:.3f}, F1 "
        f"{results['Logistic Regression + TF-IDF']['metrics']['f1']:.3f}, and ROC-AUC "
        f"{results['Logistic Regression + TF-IDF']['metrics']['roc_auc']:.3f}. Naive Bayes "
        f"reached accuracy {results['Naive Bayes + TF-IDF']['metrics']['accuracy']:.3f}, "
        f"precision {results['Naive Bayes + TF-IDF']['metrics']['precision']:.3f}, recall "
        f"{results['Naive Bayes + TF-IDF']['metrics']['recall']:.3f}, F1 "
        f"{results['Naive Bayes + TF-IDF']['metrics']['f1']:.3f}, and ROC-AUC "
        f"{results['Naive Bayes + TF-IDF']['metrics']['roc_auc']:.3f}. The higher recall of "
        f"Logistic Regression is practically valuable because missing a spam message can be "
        f"more costly than a false positive, although false positives still matter for legitimate "
        f"messages.\n\n"
        f"The Streamlit app demonstrates the saved model and TF-IDF vectorizer on single-message "
        f"screening and CSV batch review. It presents model evidence separately from heuristic "
        f"warning signals, which aligns with the binary labels available in the corpus. The core "
        f"objectives were met: source-aware data loading, preprocessing, model comparison, "
        f"validation thresholding, and a working application. The project is limited by source "
        f"overlap, in-corpus random-split evaluation, and the absence of temporal or cross-domain "
        f"testing. Future work should add independent data sources, measure generalization, "
        f"explore calibrated scores, and consider multilingual or human-reviewed labeling before "
        f"broader deployment."
    )
    document.add_paragraph(conclusion)

    document.add_heading("12. References", level=1)
    for reference in REFERENCES:
        paragraph = document.add_paragraph(reference)
        paragraph.paragraph_format.left_indent = Inches(0.28)
        paragraph.paragraph_format.first_line_indent = Inches(-0.28)
        paragraph.paragraph_format.space_after = Pt(4)
        for run in paragraph.runs:
            run.font.size = Pt(9)

    document.core_properties.title = "ScamText AI CA-3 Project Report"
    document.core_properties.author = "Nilabjo Goswami; Joshua Bara"
    document.save(REPORT_PATH)
    print(f"Created {REPORT_PATH}")
    print(f"References: {len(REFERENCES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-application-screenshot",
        action="store_true",
        help="include outputs/application_screenshot.png after it has been refreshed",
    )
    make_report(include_application_screenshot=parser.parse_args().include_application_screenshot)
