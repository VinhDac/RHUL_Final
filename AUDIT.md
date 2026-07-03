# AUDIT — Phase 0 (góc nhìn người mới, tiêu chí một-lần-đọc)

*Nguồn: workflow `phase0-newcomer-audit` (6 auditor context-sạch + 1 gộp). Đây là bản đồ lái các fix của Phase 0.5→5. Đọc cùng `Feedback.md`.*

## 6 chủ đề xuyên suốt (gốc rễ)

1. **DRIFT n_test / split — vấn đề tái lập LỚN NHẤT.** Cùng một đại lượng có nhiều giá trị mâu thuẫn: `n_test` = 100k (§3) / 20k (§5) / 10k (Appendix); finance split = 4000/200/1000 (Core:451) / 1500/200/1000 (nb) / 1500/50/1000 (nb); loan split = 1500/200/8000 / 2000/200/10000. **Không có nguồn chân lý → không con số headline nào tái lập sạch.**
2. **LIVE DOWNLOAD gây số cũ.** yfinance + ucimlrepo pull *live* mỗi lần chạy; finance trôi → §6.5 stale (64%→66%, +0.107→+0.126, +82%→+87%, lật dấu ở N=5). **Freeze data (CSV có ngày)** dập cả một lớp lỗi report≠notebook.
3. **SCAFFOLD/TODO còn sót** khiến repo *đã xong* trông như *chưa xong*: "once the lab is implemented", "You implement the cells", models.py "bodies are yours", "confirm with supervisor" (×2), §7 "not yet measured".
4. **SYMBOL-TRƯỚC-VÍ-DỤ / JARGON không giải nghĩa — lỗi một-lần-đọc chủ đạo.** Các "mồi trực giác" tốt nhất (analogy thi cử, staircase, xoay đám mây) đến *sau* phát biểu ký hiệu cùng ý; thuật ngữ tải-trọng (MLP, ReLU/logits/cross-entropy, ^GSPC, isometry, Bayes-optimal, E[], internal/external validity) thả vào lạnh. **Đây là rào chính với mục tiêu bắt buộc.**
5. **PROVENANCE code↔claim thiếu / đôi khi SAI.** header pipeline.py mô tả sai chính API của nó; Appendix A bỏ sót sklearn "model zoo", confusion-matrix loan, backtest finance; §6.1–6.3 không nêu notebook; appendix không nêu import root `snooping_backend/`; run_once/gap_kfold (trọng yếu) không có test, print-test exit 0 cả khi FAIL.
6. **NOTATION tham chiếu chéo lộn xộn** (§N vs "Phần"/"Phan" vs "SS6" vs "section N"), gồm token tiếng Việt sót (Core:392, pipeline.py:19) — examiner tiếng Anh sẽ vấp.

## 9 phần TRƯỢT tiêu chí một-lần-đọc

- **models.py** — người mới tưởng bài search trên "model zoo" 4 sklearn ở đây, thực ra là **stub chưa cài, không ai import**; search thật là MLP ở mlp.py.
- **§6.4 (MLP math)** — reader yếu ký hiệu không theo nổi ma trận/one-hot + bước log-sum-exp bỏ qua để ra `∂L/∂z = p − e_y`. (đúng nhưng cao hơn đối tượng) → khớp quyết định **đưa §6.4 xuống appendix**.
- **nb 02 (real data)** — 3 finance split khác nhau + số không khớp Core (yfinance trôi) → người mới tưởng "kết quả sai", không phải "report cũ + split thiếu định nghĩa".
- **§4 (lab design)** — rotation-invariance, RᵀR=I, wᵀx′=0, equivariant, ‖Qβ‖=‖β‖, isometry, Bayes-optimal *đến trước một bức tranh nào* → không hình thành nổi ý "vì sao coordinate-free đứng yên, tree tụt". → khớp **thêm 1 hình 2D trước đại số**.
- **§1 + §2.2** — cơ chế phát biểu bằng ký hiệu (argmax, E[], Ŝ=S+ε); "two lines of elementary probability" **hứa mà không bao giờ hiện**, trước khi có ví dụ cụ thể (mãi §3). → khớp **ví dụ trước ký hiệu**.
- **README opening** — "configuration", val-vs-test, score-là-accuracy bị giả định; câu "two amber links" vô nghĩa nếu không có màu.
- **§3 dòng 155** — ReLU/logits/cross-entropy/full-batch GD + log-uniform dồn một câu, không gloss → reader yếu ký hiệu khựng giữa Method.
- **pipeline.py** — header mô tả API *khác* code (run_once signature cũ + repeat/log_run ảo) → gọi sai → TypeError.
- **README Layout/Run** — trỏ models.py chết làm model lõi, giấu mlp.py, "once the lab is implemented" dù chạy green.

## 13 ưu tiên (đã xếp hạng)

1. **Sửa models.py end-to-end** (README Layout row + Appendix provenance + "model zoo" §5): xóa-hoặc-cài; thêm mlp.py vào Layout; ngừng trỏ code chết + giấu code sống. *(lỗi wrong-intent-một-đọc lớn nhất + gốc lỗ provenance §5)*
2. **Chốt MỘT n_test + MỘT split/dataset**, nêu một lần, dùng khắp (Core + mọi cell notebook). *(dập chủ đề tái lập chủ đạo)*
3. **Freeze finance (và loan) → CSV có ngày**; cập nhật số finance §6.5 khớp notebook. *(dập lớp lỗi report≠notebook)*
4. **Ví dụ số cụ thể vào §1 trước ký hiệu**; VIẾT "two lines of elementary probability" (hoặc demo số 2-vs-5) ở §2.2. *(vấp một-đọc lớn nhất trên claim tải-trọng)*
5. **Làm §4 sống được với reader yếu toán:** mở bằng hình 2D "xoay đám mây, khoảng cách không đổi" + "tree vẽ staircase" TRƯỚC đại số; gloss isometry/Bayes-optimal/equivariant khi dùng lần đầu.
6. **Gloss mọi thuật ngữ khi dùng lần đầu** (MLP, ReLU/logits/cross-entropy/full-batch GD, ^GSPC, E[], Ŝ, internal/external validity, nested CV, walk-forward, log-uniform).
7. **Sửa mâu thuẫn §7 "not yet measured"** (§6.1–6.3 ĐÃ đo); thêm con trỏ "reproduce from notebooks/03_extensions.ipynb" vào §6.1/6.2/6.3.
8. **Sửa header stale pipeline.py** (run_once signature thật; xóa repeat/log_run ảo); thêm test cho run_once + gap_kfold + kiểm "test chạm đúng 1 lần"; print-test **sys.exit(1) khi FAIL**.
9. **Xóa scaffold/TODO** làm repo xong trông chưa xong.
10. **Primer đầu README** (configuration, analogy thi cử + link exam_analogy.svg, hướng gap = tệ hơn); câu "amber" degrade không màu; cảnh báo reproduce cần internet + live download; results/ trống.
11. **Nêu import root `snooping_backend/`** ở Appendix A; thêm provenance row cho confusion-matrix loan (1406/2126) + backtest finance.
12. **Hiện gap_std thành error band** ở §5/§6.1/§6.2; cho ngưỡng "phẳng" bằng lời.
13. **Dọn vụn:** thống nhất §N; xóa "Phần"/"Phan"/"SS6"; xóa orphan figures/gap_synth_vs_loan.svg; caveat tree 0.849-vs-0.810; data_finance "magnitude"→signed; định nghĩa jargon trong lab.py.

## Ý nghĩa cho PLAN
Ưu tiên 1–3 (models.py · MỘT n_test/split · freeze data) là **NỀN** — phải làm **TRƯỚC Phase 1** (chạy thí nghiệm), nếu không thí nghiệm mới lại đẻ thêm drift. → thêm **Phase 0.5 — Foundation**.
