# HANDOFF — tiếp tục dựng dissertation "Data Snooping in Deep Learning"

*Dán prompt này vào đầu session mới. Nó chắt lọc kinh nghiệm từ session dựng xong §0–§2. ĐỌC KỸ trước khi làm gì.*

Bạn (Claude) tiếp quản một MSc dissertation "CS: Deep Learning" viết theo kiểu HÀNH TRÌNH (một người bình thường đi qua pipeline DL, mỗi dataset là một journey có "hidden assumption"). Session trước đã dựng xong trọn §0–§2 làm BẢN MẪU. Nhiệm vụ tiếp: **§3 Market**, rồi §4 Phone, §5 Lab, §6 Valley/§7 Summit.

## 0. Đọc trước, theo thứ tự
1. `NORTH_STAR.md` — la bàn giọng/hồn. Luật tối cao: **NHẸ**.
2. `LOAN_PLAN.md` — spec Loan + các nguyên tắc giọng đã khoá (áp cho MỌI chương).
3. `READ_ME.md` §0–§2 (tới hết Appendix C) — **bản mẫu đã chốt**, bắt chước giọng + cấu trúc.
4. `notebooks/loan.ipynb` — notebook standalone mẫu (mở xem cấu trúc + chart inline).
5. `Docs/CONSTRUCT_TOPIC_DL.md` — tầng chấm điểm (3 trụ: implement · improve ≥2 arch×≥2 opt · explain+derive+visualize).
6. memory/ tự load.

## 1. Trạng thái
- ✅ **§0/§1/§2 DỰNG LẠI XONG** (giọng cùng-khám-phá, config B, chart inline, self-contained). = KHUÔN.
- ✅ `loan.ipynb` standalone (12 cells, 6 chart inline, đọc riêng hiểu trọn).
- ✅ READ_ME §0–§2 + References (5 curated) + Appendix A (derivation) / B (loan data chi tiết) / C (code) — **TỰ ĐỦ** (đọc README riêng hiểu hết, không cần notebook).
- ✅ Refs/appendix cũ parked ở `_parked_refs_appendix.md`.
- ⏳ **§3 Market, §4 Phone, §5 Lab, §6, §7 VẪN BẢN CŨ** (config cũ + giọng cũ) → phải dựng lại.
- ⚠️ Nợ (để CUỐI): §3–§7 còn trỏ appendix đã parked (dangling); §5/§6/§7 trích số loan cũ. Reconcile + whole-report number-sweep + ẩn danh/metadata + đổ Word template ở cuối.

## 2. GIỌNG + PHƯƠNG PHÁP (phần quý nhất — chắt lọc từ rất nhiều vòng sửa)

**A. Cùng-khám-phá, KHÔNG biết trước (quan trọng nhất).** CẤM mọi "tell" báo trước: *"file this away, it'll matter" · "almost everyone would" · "the shine comes off" · "a number this smooth is suspect".* Mỗi nước đi thúc bởi cái THẤY TRƯỚC MẮT, không dùng kiến thức tương lai; để DATA (kết quả chạy thật) bung bất ngờ. Đóng vai người thực hành NGÂY THƠ.

**B. Live-run (cách làm user cực thích).** Mỗi nhịp: quyết nước đi → viết cell notebook → **CHẠY THẬT** → nhìn output thật → phản ứng thật → viết prose từ khoảnh khắc đó → mới quyết bước sau. **KHÔNG bịa/nhớ số.** Notebook = nguồn chân lý duy nhất.

**C. Cấu trúc a/b/c/d mỗi journey (đã khoá ở Loan):**
- **a — Reading the problem:** dựng TỰ TIN NGÂY THƠ; trải config ĐẦY ĐỦ rõ nhất (frame/split/scale/layers/epochs/optimizer/loss/metric); **GIẤU con số chốt-hạ** (vd. baseline). Kết "we run it, and see."
- **b — By the book:** PHỒNG (chart làm mọi thứ đẹp, "trust this number") → **"But"** → SỤP (lộ con số giấu + chart deflate) → **pulled-quote câu hỏi gây sụp đổ.**
- **c — What the number was hiding:** loại nghi phạm khác (clean control) → cú vỡ THẬT + **ví dụ đời thường dùng SỐ THẬT của bài** (kiểu bác sĩ 78%/82%, KHÔNG "one in five" trừu tượng) → **pulled-quote câu hỏi.**
- **d — The number we can trust:** cách sửa đúng → **HIDDEN ASSUMPTION** (giả định ngầm khi chạy mù recipe) → **"tin PROCESS không tin GOAL"** (số lệch không sao, tin cái CÁCH làm ra nó + các con ĐẾM thô) → **pulled-quote câu hỏi mở sang chương sau.**

**D. Quy ước trình bày:** câu hỏi trọng tâm = **pulled-quote** (blockquote đậm); câu key phụ = bold/italic THƯA tay; **MỖI MỤC KẾT BẰNG CÂU HỎI NGHI VẤN** (không phải câu chốt); diễn giải KEY trực quan + số THẬT + highlight bold; chart "phồng" zoom-in vs "sụp" zoom-từ-0 = bài học ngầm. **TUYỆT ĐỐI KHÔNG em-dash (—)**.

**E. Hidden assumption = đích mỗi journey.** Mỗi bước recipe làm mù giấu một giả định data phá vỡ. Loan = MEASURE giả định "lớp cân bằng, lỗi giá ngang." Nối vào thesis cả bài.

**F. Self-contained kép.** README đọc riêng hiểu hết (appendix chứa data/derivation THẬT, không chỉ "xem notebook"); notebook đọc riêng hiểu hết (markdown giàu + chart inline).

**G. Mode C.** Mỗi lựa chọn logic → đề 2–3 phương án → **USER QUYẾT** → viết mẩu nhỏ → soi la bàn → chốt → bước sau. CHẬM, mẩu nhỏ. **KHÔNG commit git.** Không tự quyết arc.

## 3. KỸ THUẬT
- **Python:** dùng `python` HỆ THỐNG (3.14, có numpy 2.4.3 + matplotlib 3.10.8). **`.venv` HỎNG** (trỏ Python 3.11 đã xoá) — ĐỪNG dùng.
- **Dựng notebook:** viết 1 script builder ở scratchpad (nbformat + `ExecutePreprocessor(kernel_name="python3")`, `metadata={"path": ".../notebooks"}`), construct tất cả cell (markdown giàu + code) rồi execute để lưu output THẬT; re-run mỗi lần đổi. *(Builder script ở scratchpad là tạm/không persist sang session mới — nhưng notebook + repo file thì persist; cứ dựng builder mới theo pattern này.)* Warning zmq Proactor = vô hại.
- **Chart inline:** đầu code cell dùng matplotlib đặt `%matplotlib inline` + `%config InlineBackend.figure_format='svg'`; mỗi hình: `fig.savefig(f"{FIG}/x.svg"); plt.show()` (savefig để README có file, show để notebook hiển thị inline). `FIG = "figures" if os.path.isdir("figures") else "../figures"`.
- **Config (mẫu Loan; dùng SHAPE tương tự cho chương khác trừ khi chương cần khác — Market cần chronological split + rolling scaler):** train/val/test 60/20/20 (val để dành §5), standardize trên train, MLP feats→16 ReLU→2 softmax, cross-entropy, plain GD lr 0.3, 300 epoch, seeds 0–4. **GUARDRAIL:** giữ baseline NGÂY THƠ (không cân-bằng-lớp; không momentum ở §2–4 — momentum để §5 cải thiện F2).
- **Palette:** accent `#3a6ea5`, muted `#b8c0cc`/`#9db8d6`, đỏ `#c85a52`, spine sạch, grid `#eef1f4`.
- **Số Loan (config B — để reconcile §5/§6/§7 sau):** acc 0.8195(seed0)/0.817(mean) · do-nothing 0.7788 · defaulters 1353→caught 472/missed 881 (recall 0.349) · payers 4647→cleared 4445/flagged 202 · balanced 0.646 vs 0.500 (gap balanced 0.146 vs accuracy 0.038).
- **Figures Loan đã có:** loan_client, loan_pipeline, loan_learning, loan_stability, loan_baseline(waffle), loan_deflate, loan_recall, loan_balanced.

## 4. NHIỆM VỤ NGAY: §3 Market
- Áp KHUÔN Loan (a/b/c/d · live-run · chart inline · câu kết = câu hỏi · self-contained · notebook `market.ipynb` standalone).
- Market có **HAI bẫy** (theo CHAPTER_MAP + §3 cũ): (1) **rò split** — shuffle thổi điểm vì các dòng near-twin (X[0], X[1] chung 4/5 giá trị); (2) **thang trôi** — frozen scaler vỡ khi đổi feature sang "points" trôi theo index (677→7610). Key question kiểu *"an edge this easy? what's leaking in?"*. Data `data/gspc_2026-07-03.csv` (một cột giá S&P).
- **BẮT ĐẦU:** đọc §3 cũ trong READ_ME + `market.ipynb` cũ để nắm số/cấu trúc/2-bẫy → rồi hỏi user cách vào (Mode C) → **CHẠY THẬT trước khi viết một chữ.** *Chú ý: Market khác Loan — 2 bẫy nằm ở bước SPLIT + SCALE (không phải MEASURE); mỗi bẫy có hidden assumption riêng.*

## 5. Nhắc cuối
Luôn hỏi user trước mỗi lựa chọn logic. Soi `NORTH_STAR.md` + `LOAN_PLAN.md` mỗi mẩu. User cực chú trọng: **giọng nhẹ · cùng-khám-phá THẬT (không biết trước) · số THẬT từ notebook chạy live · self-contained · mỗi mục kết bằng câu hỏi · không em-dash.**
