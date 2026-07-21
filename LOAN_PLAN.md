# LOAN (§2) — note làm lại chỉn chu

*Working note cho việc dựng lại §2 Loan (prose + `loan.ipynb`) theo giọng CÙNG-KHÁM-PHÁ và **config B**. Chốt 2026-07-21. La bàn giọng: `NORTH_STAR.md`. Đây là spec riêng của Loan; khi vênh, NORTH_STAR + Topic thắng.*

---

## 0. Vai của Loan trong cả bài
- Loan = **ĐỐI CHỨNG SẠCH**. Bẫy duy nhất = **thước đo** (accuracy che recall kém trên nhóm vỡ nợ hiếm). KHÔNG rò, KHÔNG trôi, KHÔNG snoop.
- **KEY QUESTION** (pulled-quote §2c): *"Of the clients who actually did default, how many did the model catch?"*
- **Trọng tâm thu được:** data + method hoàn hảo, số đo chính xác tuyệt đối, **vẫn trả lời sai câu hỏi**. Bẫy nằm ở cái ta *chọn để đo*.

## 1. Nguyên tắc giọng (từ cú sửa "đừng biết trước")
- Người đồng hành, KHÔNG phải giáo sư. Cùng khám phá thật.
- **CẤM tell biết-trước:** không "file this away, it'll matter"; không "almost everyone would"; không "a number that smooth is worth suspecting"; không "the shine comes off". Mỗi nước đi thúc bởi cái thấy TRƯỚC MẮT; để **DATA** bung bất ngờ, không phải người kể báo trước.
- Nhẹ (chuẩn §0). Không em-dash.
- **Quy ước trình bày:** câu hỏi trọng tâm = pulled-quote (blockquote đậm); câu key phụ = bold/italic thưa tay; pipeline = list đánh số + figure nâng cấp.
- **Câu KẾT mỗi mục = LUÔN là câu hỏi nghi vấn** (không phải câu chốt/statement) — mỗi beat khép bằng một mối nghi đẩy tới beat sau. (Áp cho §2b, §2c; §2d kết bằng câu hỏi mở sang §3.)
- **Diễn giải KEY phải trực quan, đời thường** (vd. ví dụ bác sĩ 100 bệnh nhân) + highlight bold; ưu tiên số đếm cụ thể hơn tỉ lệ trừu tượng; chart "phồng" zoom-in vs "sụp" zoom-from-0 là bài học ngầm.

## 2. §2a — CHỐT: dựng tự tin ngây thơ, KHÔNG kết quả
- **Nhiệm vụ:** nêu vấn đề + **trải phương án cụ thể, rõ nhất có thể** → độc giả *hiểu* + *tin ngây thơ* "chuẩn bị bài bản thế này thì cứ áp là xong, khỏi nghĩ nhiều".
- **Nội dung theo trình tự:** bài toán → đọc 1 client → mục tiêu → **CONFIG đầy đủ** (frame / split / scale / model / forward-backward / optimizer / loss / measure) rõ hết → **sàn 0.779** (data-fact, ở lại) → kỳ vọng ("we run it, and see").
- **KHÔNG một số KẾT QUẢ nào** (điểm số của model). Số **CONFIG** thì có (là phần "cụ thể rõ").
- **0.779** = tỉ lệ khách KHÔNG vỡ nợ (77.9%) = accuracy của "đoán lớp đông". **Hạt giống plot-twist**, bung ở §2c.
- Emotional arc: §2a = xây lòng tin → b/c = dẫn tin tưởng rồi **bẻ gãy** ("nó chẳng đúng đâu").

## 3. Config (B — CHẠY LẠI; baseline ngây thơ nhưng làm cực sạch)
- **Split:** train / validation / test, tách 1 lần từ đầu (khớp pipeline §1). §2 fit trên train, báo cáo trên test. **Validation KHÔNG dùng trong §2** (Loan không chọn-model/early-stop); nó để đủ pipeline, và search thật sống ở §5.
- **Scale:** standardize, fit trên train only.
- **Model:** MLP, 1 lớp ẩn 16 nút ReLU → softmax(2 lớp). Loss **cross-entropy**.
- **Optimizer:** **gradient descent thường** (KHÔNG momentum — momentum dành cho §5). lr + epoch cố định, **seed cố định**, báo cáo qua vài seed.
- **Measure:** accuracy, so sàn 0.779.
- **GUARDRAIL (sống còn):** KHÔNG cân-bằng-lớp / reweight / đổi threshold. Ngây thơ *chính là* cái bẫy — "sửa" imbalance trong config = mất bài học.

### Vì sao KHÔNG làm §2 tinh vi hơn (couplings)
- Thêm val + early-stopping vào §2 = kéo cơ chế **snoop** của §5 vào sớm → §2 hết "sạch".
- §5 phải **cải thiện** baseline của §2 (1 lớp + GD → 2 lớp + momentum). Nếu §2 đã tối ưu, §5 không còn gì để cải thiện.

## 4. Số phải KIỂM sau khi chạy lại (mạch phải giữ; vỡ thì chỉnh TRƯỚC khi viết prose)
- [ ] Split không quan trọng: shuffle nhiều lần + stratified **đồng thuận** trong nhiễu → đối chứng sạch. (Cắt theo thứ tự file = artifact nhẹ, nhánh bị tỉa.)
- [ ] Accuracy chỉ **nhỉnh** hơn sàn 0.779 một chút.
- [ ] Recall trên nhóm vỡ nợ **kém** (~1/3 bắt được) → cái bẫy.
- [ ] Balanced accuracy ~0.6x → số trung thực để giữ.
- *Giá trị chính xác: CHỜ chạy lại. Notebook = nguồn chân lý duy nhất.*

## 5. §2b/c/d (cùng-khám-phá; kết quả ở đây)
- **§2b — làm + nhẹ nhõm:** chạy by-the-book → ra accuracy. Điều tra "split có quan trọng không" — chỗ quan sát **"không date, không id"** NẢY RA tự nhiên khi đứng trước việc chia (không cất sẵn ở 2a). Các cắt đồng thuận → nhẹ nhõm, tự tin còn nguyên.
- **§2c — cú vỡ:** câu hỏi cứng (KEY QUESTION, pulled-quote) đến từ **tình huống tức thời** (không phải khôn-ngoan-tác-giả) → đếm bắt/sổng → accuracy cao nhưng recall kém; **do-nothing cũng ~0.779** → plot-twist 0.779.
- **§2d — chiêm nghiệm + sửa:** không rig, đo hoàn hảo, sai câu hỏi → balanced accuracy → **điểm cân bằng** (giữ số trung thực, bỏ số phỉnh).

## 6. Coupling / ripple (reconcile SAU khi chạy lại)
- **§5 Lab** reuse loan features + tham chiếu **balanced accuracy** của §2 (baseline mà honest search F2 cải thiện). Cập nhật số kiểu "0.644 → 0.650".
- **§6 Valley, §7 Summit** recap số §2. Cập nhật.

## 7. Nâng cấp `loan.ipynb` — LÀM TRƯỚC prose
- Config mới; in **CONFIG block** rõ (arch / epochs / lr / loss / optimizer / split / seed) = nguồn chân lý cho prose.
- Cấu trúc lại mirror a/b/c/d.
- Regen figure (pipeline loan-specific + confusion) theo gu nâng cấp (badge số, accent nhẹ, print-safe).
- Tái lập được (seed cố định); in transform từng dòng.

## 8. Thứ tự thực thi
1. [x] Note này.
2. [ ] Nâng cấp `loan.ipynb` → chạy lại → **kiểm mạch (mục 4)** → báo số mới cho user.
3. [ ] Viết **§2a** (config rõ hết, 0 kết quả) + nhúng figure pipeline.
4. [ ] **§2b/c/d** với số mới.
5. [ ] Reconcile **§5 / §6 / §7**.
