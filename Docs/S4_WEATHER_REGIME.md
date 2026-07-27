# §4 (bản mới đang cân nhắc) — DỰ BÁO THỜI TIẾT THỊ TRƯỜNG qua REGIME

*Parking doc, 2026-07-26. **QUYẾT (2026-07-26): đây là NHÁNH SONG SONG — CHƯA bỏ §4-VaR; prose §4 a/b/c hiện có giữ nguyên.** CHƯA viết vào READ_ME. Doc-làm-việc (tiếng Việt); prose README sẽ dịch sang English sau.*

> 🔴 **PIVOT LỚN (Beat 29, 2026-07-26): "THỜI TIẾT" = CHỈ VOLATILITY (êm / thường / bão), MỘT trục honest. BỎ trục DIRECTION làm feature.** Bước 0–1 + audit chứng minh direction ≈ nhiễu + artifact vi-cấu-trúc, vác nhiều hidden-assumption nhất, và chạm guardrail "hướng = đồng xu" của §3. **Cuộc điều tra direction được GIỮ làm KEY FINDING** (minh chứng snooping sống: một trục nghe rất hợp lý, chuẩn hoá+band lên thì *trông có nghĩa*, test tử tế mới lộ là nhiễu). ⇒ Mọi mục "2 trục / 6-regime" bên dưới là **ý GỐC, nay SUPERSEDED** thành vol-only. Bài học đầy đủ ở **§13**.

---

## 0. Ý một dòng
Mọi thị trường đều có "thời tiết". Ta phân loại thời tiết đó bằng **hai thang — Memory (trí nhớ giá) × Volatility (độ động)** → 6 regime. Ướm cho **MỌI dataset đã normalize**, đo **theo tuần**, **gộp các tuần liền kề cùng loại** thành đoạn regime lớn → ra **LỊCH SỬ THỜI TIẾT** của từng symbol, và bức tranh thời tiết **TOÀN THỊ TRƯỜNG** qua các symbol đại diện. **Không cố hoàn hảo** — như dự báo thời tiết thật.

## 1. Vì sao (mạch nối tới §3)
- §3 dựng "mai busy/calm": đúng, nhưng vô dụng, vì không biết dùng để làm gì.
- Bài học: **không biết context/mục đích thì không dựng nổi bài toán sạch.**
- §4: cho nó một mục đích thật = một **bản tin thời tiết thị trường** mà người ta đọc được.

## 2. Hai thang + 6 regime (từ bảng của user)

| Regime | Memory | Vol | Nghĩa |
|---|---|---|---|
| **R1 Slow Trend** | Directional | Low | hướng rõ, vol thấp, trend sạch/chậm, ít hoảng loạn |
| **R2 Crisis Trend** | Directional | High | hướng rõ, vol cao, gap, dễ biến động lớn |
| **R3 Tight Range** | Range/Revert | Low | hay quay lại vùng cũ, vol thấp, dao động trong biên |
| **R4 Volatile Chop** | Range/Revert | High | hay đảo ngược, vol cao, rung lắc, khó theo trend |
| **R5 Drift** | Random | Low | memory không rõ, vol thấp, giá trôi nhẹ, tín hiệu yếu |
| **R6 Noise** | Random | High | memory không rõ, vol cao, nhiều nhiễu, khó làm nhãn tốt |

17 symbol mẫu (từ tool user, để tham chiếu): EURUSD·R3, US100·R1, US500·R1, BTCUSD·R1, AUDUSD·R3, XAGUSD·R2, XAUUSD·R2, USDJPY·R1, ETHUSD·R2, GBPUSD·R2, US2000·R3, NZDUSD·R3, USDCHF·R2, USDCAD·R3, US30·R1, UKOIL·R5, USOIL·R5.

## 3. Phương pháp (bản phác — chờ chốt)

**Tầng A — MÔ TẢ (làm lịch sử regime; đây là phần honest):**
- Mỗi symbol đã normalize → chia theo **TUẦN**.
- Mỗi tuần đo 2 toạ độ: (i) **Volatility** tuần (thấp/cao), (ii) **Memory** tuần (directional / range / random) → gán 1 regime.
- **Gộp các tuần liền kề cùng regime** thành một "đoạn regime lớn" → chuỗi các đoạn = **lịch sử thời tiết**.

**Tầng B — DỰ BÁO (đoán chuyển giao; khó, KHÔNG ép hoàn hảo):**
- Cái đáng giá = **ĐOẠN CHUYỂN GIAO** (regime sắp đổi sang loại khác).
- Coi trọng đoạn này; chấp nhận không hoàn hảo (đúng tinh thần thời tiết).

**Kết quả kỳ vọng:**
- Lịch sử regime của TỪNG dataset.
- Bức tranh thời tiết TOÀN THỊ TRƯỜNG qua các symbol đại diện.

## 4. Ranh giới HONEST (cái PHẢI làm kĩ để đúng đề tài, không phải để khoe)
- **Thang VOLATILITY = vững.** §3 đã chứng minh vol dồn cụm, có tín hiệu thật (autocorr +0.287). Dự báo trên trục này là honest.
- **Thang MEMORY/HƯỚNG = vùng snoop-prone.** §3 chứng minh hướng/trend gần như đồng xu. → trục này nên dùng để **MÔ TẢ quá khứ**, rất **dè dặt khi DỰ BÁO**. Tô nó thành "đoán được hướng" là phản §3.
- **Rò rỉ nhãn.** Đo regime tuần bằng dữ liệu TRONG tuần đó = OK cho MÔ TẢ. Nhưng nếu lấy nhãn có liếc tương lai làm **target dự báo** thì rò rỉ. Phải tách bạch A và B.
- **Mọi NÚM là snooping-knob → phải khai báo trước:** độ dài cửa sổ (1 tuần? mấy tuần?), ngưỡng "vol cao" (cố định / phân vị / theo từng symbol), cách đo "memory" (autocorr / Hurst / % đảo chiều), luật gộp đoạn. Mỗi lựa chọn đổi kết quả → cắm trước, khai báo, KHÔNG dò tới khi đẹp.
- **"Không cố hoàn hảo" = tính năng, không phải lỗi.** Một dự báo thời tiết honest thì khiêm tốn nhất ở chỗ chuyển giao.

## 5. Nối thesis "Data Snooping in Deep Learning"
- Vẫn là bài snooping: *"thêm asset / thêm núm có làm số đẹp lên, và có tin được không?"* là đúng câu hỏi snooping.
- **Phép thử phổ quát chống snooping:** một regime/tín hiệu THẬT thì xuất hiện ở **nhiều symbol**; snoop thì chỉ ở một chỗ rồi **chết khi đổi symbol**. → đa-dataset thành một PHÉP THỬ, không phải màn phô diễn.
- Chuyển giao regime = tái ngộ bài học **drift/era** của §3 (thế giới đổi chế độ).

## 6. Data (thực tế phải đối mặt)
- ✅ ĐÃ LẤY (2026-07-26, PA-B công khai): **17 symbol daily OHLCV** từ **Yahoo Finance v8 chart API** (stdlib urllib, không cần cài gì). Script tái lập: `data/fetch_panel.py` (đóng băng end=2026-07-26). File: `data/panel/<SYMBOL>.csv` + `data/panel/_manifest.csv`. Đã validate: ngày tăng dần, không trùng, giá dương (trừ 1 điểm USOIL, xem dưới).
- Mapping (broker→Yahoo): index US500=^GSPC, US100=^NDX, US2000=^RUT, US30=^DJI; FX =X; crypto BTC-USD/ETH-USD; **hàng hoá dùng FUTURES liên tục làm proxy**: XAUUSD=GC=F, XAGUSD=SI=F, UKOIL=BZ=F, USOIL=CL=F (⚠️ chờ user duyệt spot-vs-futures).
- ⚠️ **Lịch sử LỆCH NHAU rất nhiều:** US500 từ 1970 (14260 dòng) nhưng ETH chỉ từ 2017 (3182), BTC 2014, AUDUSD 2006, UKOIL 2007. → không so được "era" trên toàn lịch sử; cần **cửa sổ chung** (crypto ép mốc ~2017+) hoặc xử theo từng symbol. (quyết ở #4 dưới)
- ⚠️ **USOIL có giá ÂM −37.63 ngày 2020-04-20** (sự kiện WTI âm, thật). Phá phép return khi đổi dấu → phải drop/xử riêng ngày đó.

## 7. Quyết định CÒN MỞ (mẩu tiếp, bàn từng cái)
1. ✅ CHỐT (2026-07-26): **Memory = tự-tương-quan return có dấu (lag-1 trong cửa sổ)** — dương=Directional / âm=Revert / ~0=Random. Chọn vì đơn giản-sạch, nối thẳng §3; loại Hurst (đỏng đảnh + thêm núm R/S vs DFA).
2. **Vol đo bằng gì:** đề xuất (chờ chốt) **realized vol = trung bình |return| trong cửa sổ** (khớp §3, bền đuôi; loại EWMA/GARCH/Parkinson vì thêm núm/giả định). — Còn **ngưỡng "cao/thấp" cắt ở đâu** (cố định / phân vị / riêng từng symbol) để chốt chung với ngưỡng dấu của Memory.
3. **Cửa sổ = đúng 1 tuần?** Luật gộp đoạn ra sao? (Tần suất ĐÃ chốt = DAILY — intraday không khả thi cho panel lịch-sử-dài; daily-vs-intraday chỉ làm robustness phụ trên 1–2 symbol đoạn gần.)
4. ✅ CHỐT nguồn+tập (2026-07-26): **17 symbol qua Yahoo daily** (xem §6). CÒN MỞ: (a) duyệt proxy futures cho 4 hàng hoá; (b) **cửa sổ CHUNG** để so era (crypto ép ~2017+) hay xử per-symbol; (c) xử ngày USOIL âm.
5. **Phần DL nằm ở đâu** (classifier regime? dự báo chuyển giao?) và **bung tới đâu** để ăn điểm topic (≥2 arch, ≥2 optimizer, search, convergence, derivations, survey) mà KHÔNG phá honest?
6. ✅ ĐÃ CHỐT (2026-07-26): **nhánh SONG SONG, chưa bỏ §4-VaR.** (Nghĩa là ta phát triển nhánh này riêng, so kè sau; không đụng prose §4 a/b/c.)

## 8. Rủi ro với cấu trúc đã khoá (ghi lại để không quên, không phải để cản)
- Cấu trúc hai-hồi (user-locked 2026-07-24): "**cùng MỘT bài toán, một thị trường**, tay mơ → bậc thầy". Bản này mở sang **nhiều symbol + trục hướng** → nếu đi xa quá, mất cú "giải lại chính bài §3" và chạm guardrail "size only, never direction".
- Cách giữ an toàn (đã bàn): giữ **lõi dự báo ở trục độ-lớn/vol**; trục memory chỉ để **mô tả**; đa-symbol làm **phép thử chuyển-giao/robustness**, không phải để tuyên bố đoán được hướng.

## 9. PHÁT HIỆN AUDIT + RÀNG BUỘC (Beat 20, 2026-07-26)
> ⚠️ §0–§3 ở trên là **bản phác GỐC** (6-regime, tên "Memory", đo tuần signed autocorr). Đang **định hình lại** thành pipeline sạch (đổi tên Direction; kháng outlier; per-symbol PAST-ONLY normalize; DL = dự báo regime kế; test held-out time+symbol). Flow mới ghi ở **§10** sau khi tác giả duyệt.

Audit (đọc lại code scratch `notebooks/weather_regime_scratch.ipynb` + verify bằng số). Xếp theo mức nặng:

**🔴 Đã kiểm bằng số:**
- **F1. "Tín hiệu Direction" (Beat 19) là OUTLIER ARTIFACT — RÚT LẠI.** autocorr không kháng outlier: EURUSD raw ac −0.175 → winsor(1/99) −0.037, rank −0.043 (do MỘT ngày 2008-12-08 +16%); USDJPY −0.124→−0.035; US500 đổi DẤU vì Black Monday 1987. ⇒ sau khi kháng outlier, **direction ≈ nhiễu gần hết symbol** (củng cố §3). **RÀNG BUỘC:** mọi phép đo KHÁNG OUTLIER (winsorize/rank); direction phải re-test sau khi sạch trước khi tin.
- **F2. Data có BAR LỖI:** EURUSD +16% (2008-12-08), silver −37.6% (2026-01-30), USOIL −37.63/60%, roll futures. **RÀNG BUỘC:** clean/winsorize TRƯỚC mọi phép đo.
- **F3. Cửa sổ KHÔNG đồng nhất calendar:** crypto 7 bar/tuần (khác ~5); "252 bar" = 8.3 tháng cho crypto vs ~12 tháng khác. **RÀNG BUỘC:** thống nhất cửa sổ theo calendar (hoặc ghi rõ per-bar).

**🟠 Lỗi logic/toán chắc chắn:**
- **F4. Full-sample normalization = RÒ RỈ lookahead** vào target (mean/std/rank dùng cả dữ liệu sau kỳ). **RÀNG BUỘC:** normalize/threshold PAST-ONLY (expanding) khi làm target DL. (Mô tả thuần thì full-sample OK.)
- **F5. Weekly readings CHỒNG nhau** (cửa sổ dài chồng ~99%) → mẫu không độc lập → train/test naive thổi phồng. **RÀNG BUỘC:** DL dùng kỳ không chồng / tính đến tự-tương-quan.
- **F6. Sig-test dùng sàn lý thuyết 1/√n thay null thực** → thổi phồng significance (US2000 "42%" ảo một phần). **RÀNG BUỘC:** empirical null (multi-shuffle).

**🟡 Giả định/knob đã ghi nhận:** z-score trên vol lệch phải → band bất đối xứng; ann.vol ×√252 cho crypto (nên √365); drop bar lỗi → return ghép nhiều ngày; **raw** (không adjusted) close; futures continuous có roll jumps; **TOP-LEVEL:** chính mô hình "2 trục vol×direction + band low/med/high" là MỘT giả định lớn, chưa chứng minh là cách cắt "thời tiết" đúng.

## 10. PIPELINE SẠCH — flow đã DUYỆT (Beat 21, 2026-07-26), 7 bước
Mỗi bước gấp sẵn ràng buộc audit (F1–F6):
- **0. Data & Clean:** nạp 17 symbol → log return → **kháng outlier (winsorize)** + xử bar lỗi (F2) + **calendar đồng nhất** (F3). Ra: return sạch per symbol.
- **1. Đo trục VOL (robust)** = |ret| TB. [✅ ĐÃ LÀM: điều tra direction → **BỎ**, xem PIVOT §top + §13.] Weather = 1 trục vol.
- **2. Gán nhãn VOL:** chuẩn hoá per-symbol **ROLLING 52-TUẦN PAST-ONLY** (F4 + non-stationarity). ⚠️ **Window length LOAD-BEARING** (agreement chỉ **75%** giữa 26/52/104, khác winsor 97% / method 91–94%) → **chọn 52 theo NGUYÊN LÝ adaptivity**: 26 nhảy + "ăn" bão kéo dài, 104 chậm 2 năm (~phản bội lý do chọn rolling). Cắm trước, chống snooping. → band **êm/thường/bão**. Cách chuẩn hoá = **z** (Vấn đề 2 ✅: 4 method z/log-z/percentile/robust-MAD trùng 89–94% → KHÔNG load-bearing → chọn z đơn giản; MAD là bản robust tương đương). Số mức + ngưỡng = **3 mức (êm/thường/bão) tại z = ±0.5** (Vấn đề 3 ✅: 5 mức nhấp nháy 2.5wk + lệch → loại; 2 mức thô; ngưỡng LOAD-BEARING [bão 28%/24%/19% ở ±0.5/±0.7/±1.0], ±1.0 để "thường" nuốt 60% → loại; **±0.5 cho lớp CÂN BẰNG nhất** = tốt nhất cho DL target + calibration). ✅ **BƯỚC 2 XONG:** vol=TB\|ret\| 20 ngày · baseline rolling 52 tuần · z · 3 mức ±0.5.
- **3. Dataset forecast:** feature QUÁ KHỨ → target band **t+1**; split **held-out TIME + SYMBOL**; **kỳ không chồng** (F5). ✅ **Horizon = 1 tuần** (lab: persistence 75.3% vs base-rate 39.9% = **+35.4% tín hiệu**; tàn nhanh 24.6/10.7/4.5% ở 2/4/8 tuần). ⚠️ Hai điều honest: (i) forecast KHẢ THI (khác direction ≈ nhiễu); (ii) **baseline persistence CAO ~75% → DL phải vượt 75%** = báo trước "phức tạp mua ít". Còn lab: FEATURE · SPLIT · baseline.
- **4. Model DL pooled:** ≥2 arch, ≥2 optimizer, search (điểm topic); baseline persistence/logistic.
- **5. Đánh giá:** vs baseline · calibration · held-out time+symbol · **transfer test**.
- **6. Sản phẩm + Self-audit** thường trực (như Beat 20).

## 11. KEY VISUALS + KEY CONCLUSIONS (CHỐT a, Beat 24, 2026-07-26)
**Visuals:**
- KV1. Bản đồ 2 trục Direction × Volatility (lưới 6/9 regime) — mỏ neo khái niệm. *(vẽ được ngay)*
- KV2. Winsor before/after (glitch tame, khủng hoảng sống). *(có số)*
- KV3. Direction là nhiễu: cú sập −0.175→−0.038 + real-vs-null. *(có số)*
- KV4. Timeline regime 1–2 symbol (vol nhanh, direction nền chậm). *(pending Bước 1–2)*
- KV5. Transfer test + DL vs baseline. *(pending Bước 4–5)*

**Conclusions:**
- KC1. Vol = trục mạnh phổ quát → weather = vol; **Direction ĐÃ BỎ** (yếu/nhiễu/microstructure, §13) — §3 tái xác nhận trên 17 chợ.
- KC2. Chuẩn hoá+band làm cả NHIỄU THUẦN trông có nghĩa → phải test raw-vs-null, đừng tin nhãn đẹp.
- KC3. Một ngày glitch giả ra "tín hiệu direction" → làm sạch kháng-outlier là bắt buộc.
- KC4. "Thời tiết phổ quát" honest = model chung + chuẩn hoá per-symbol + test held-out-symbol; giá trị dự báo khiêm tốn, vol dẫn, phức tạp mua ít.

## 12. TIMEZONE — cách xử (Beat 25, 2026-07-26)
- **Vấn đề:** symbol đóng cửa giờ khác nhau (FX ~5pm NY, index ~4pm ET, crypto UTC, futures settlement lệch) → "cùng một ngày" KHÔNG đồng thời.
- **Chỉ cắn** với **cross-asset contemporaneous feature** (dùng dữ liệu symbol khác "cùng ngày" để dự báo symbol này → có thể rò rỉ tương lai). **Own-symbol sequence tự nhất quán** (mọi bar của một symbol đóng cùng giờ) → KHÔNG sao.
- **QUYẾT:** feature = **OWN-SYMBOL only** (v1). Model **pooled shared-weights** + **held-out-symbol transfer** vẫn chạy đủ → **timezone được DESIGN AWAY**, không phải vá.
- Nếu sau thêm cross-asset "market-wide" feature: **lag ≥1 ngày** (leak-proof bất kể thứ tự giờ đóng). Caveat: giờ đóng chính xác của Yahoo chưa verify; own-symbol làm nó moot cho core forecast.

## 13. BÀI HỌC DIRECTION (saved, Beat 29) — điểm sáng của journey
**Quyết (2026-07-26): thời tiết = CHỈ volatility (êm/thường/bão). Direction BỎ khỏi model, GIỮ làm key finding.**

**Ta đã thử gì:** ý gốc = "thời tiết thị trường" hai trục — Volatility (độ động) × Direction (trend/revert/random, đổi tên từ "Memory"). Trực giác rất hợp lý; tool của tác giả cũng gán mỗi symbol một regime 2-trục.

**Ta phát hiện gì (bằng số, Bước 0–1 + audit):**
1. Chuẩn hoá + band làm **cả nhiễu thuần** cũng ra low/med/high đẹp (31/38/31) → phân phối đẹp KHÔNG chứng minh gì. Phải test **raw vs null**.
2. autocorr KHÔNG kháng outlier: "tín hiệu direction" mạnh (EURUSD −0.175) hoá ra do **một ngày glitch +16%** → về −0.038 khi kháng outlier. (Claude từng bị lừa ở Beat 19.)
3. Sau khi sạch: autocorr return ≈ 0; vol clustering +0.181 (mọi symbol) → **vol mạnh gấp ~9× direction**.
4. Null đúng (sign-shuffle giữ vol clustering) → chỉ 9/17 còn "significant", magnitude 0.03–0.06 (**kinh tế không đáng**); phần mạnh nhất = chỉ số cổ phiếu theo thứ tự small-cap→large-cap = **chữ ký non-synchronous trading (artifact vi-cấu-trúc)**, không phải memory dự báo được.

**Vì sao BỎ:** direction yếu ~9×, phần "signal" là microstructure, vác **nhiều hidden-assumption nhất** (nhạy outlier, era-drift, chọn null, timezone nếu cross-asset), và **chạm guardrail** "hướng = đồng xu" của §3. Giữ nó = rước đúng cái snooping luận văn chống.

**Vì sao GIỮ cuộc điều tra:** nó là **minh chứng snooping sống động nhất** của branch — một trục nghe hợp lý, band lên trông có nghĩa, chỉ test tử tế mới lộ là nhiễu + artifact. Thành **KC2 + KC3**, nối thẳng §3.

**Meta-lesson:** quy trình cứu ta = **đo → nghi → test vs null ĐÚNG → audit hidden-assumption → bỏ nếu không thật.** Chính là quy trình honest mà Hồi kết sẽ đặt tên.

## 14. OVERLAP CATCH → NHỊP THÁNG (A) (Beat 37–38, 2026-07-26)
**CATCH (F5) — bắt được nhờ tác giả đòi lab feature:** lab horizon/feature bị **OVERLAP LEAK**. Vol 20 ngày TRƯỢT + bước TUẦN → vol[t] và vol[t+1] chung **75% dữ liệu** → persistence 75% / R²=0.79 là **ARTIFACT** (đọc lại chính data, không phải dự báo). Re-measure KHÔNG chồng: 1-tuần THẬT chỉ **+5.7%** (yếu ~ direction), 1-tháng THẬT **+17.3%**.

**QUYẾT A (tác giả, nhịp THÁNG non-overlap) — cụ thể:**
1. Cắt mỗi symbol thành **khối 20 ngày giao dịch KHÔNG chồng** (khối 1 = ngày 1–20, khối 2 = 21–40...).
2. vol khối = **TB|ret| của đúng 20 ngày đó** (vẫn thước vol đã khoá, chỉ NHẢY thay vì TRƯỢT).
3. Chuẩn hoá **z past-only**, baseline = **24 khối trước (2 năm)** ✅ (lab 12/24/36: baseline dài→tín hiệu mạnh+ổn nhưng kém adaptive [tradeoff như window-length]; 12 quá ít điểm→z nhiễu→tín hiệu yếu +9.3%; 36 kém adaptive & giống 24 tới 84%; **24 = z đáng-tin + adaptive + tín hiệu +14.2%**, cắm trước chống snooping).
4. Band 3 mức ±0.5 (đã khoá) → **mỗi tháng MỘT nhãn**.
5. Dự báo nhãn **tháng TỚI** từ tháng trước → **không leak**; tín hiệu thật +17.3%; **baseline persistence ~58% → DL phải vượt 58%**.
6. Split: tháng cũ→train, tháng mới→test + held-out symbol → mẫu ĐỘC LẬP → split SẠCH.

Trade-off honest: **~5000 mẫu sạch** (1/5 số tuần); baseline 12 khối (12 điểm) ít hơn 52 tuần → **đang lab N** (đừng để trôi như window length).
⇒ **SUPERSEDES §10 bước-2 "rolling 52 tuần" + bước-3 "horizon 1 tuần".**

**FEATURE (Beat 39) = [current vol-z, lag1]** own-symbol. Lab (monthly non-overlap): current **R²=0.27** mang gần hết tín hiệu; lag REDUNDANT vì vol đổi chậm (tháng trước ≈ tháng này) → +lag1 chỉ **+0.011**, lag2/lag3/trend **+0** → BỎ. (R²=0.79 tuần cũ = overlap artifact, đã xác nhận.) Nguyên tắc: chỉ nhận feature lab chứng minh có ích → tối giản, tránh chỗ overfit.

## 15. LIMITATIONS + BÁO ĐỘNG (cắm TRƯỚC DL, Beat 40) — lie-detector kiểu §4
**AUDIT trước DL: leak-placebo PASS** (real persistence 54.8% → shuffled-target 34.5% = may rủi → không có feature→target leak). Pipeline sạch. Block-size 20 là tradeoff (signal 8.5/14.2/16.4% ở 10/20/40) nhưng 20=tháng tự nhiên, cắm bằng nguyên lý.

**KHI NÀO SAI / KHÔNG LÀM ĐƯỢC:**
1. Chuyển giao: persistence fail mạnh nhất ĐÚNG lúc bão vừa nổ (lúc cần nhất).
2. Tín hiệu yếu (+14%) → sai ~40–45%, là *tilt xác suất* không phải chắc chắn.
3. Đổi chế độ >2 năm → baseline rolling-24 trễ tới 2 năm.
4. Nhãn TƯƠNG ĐỐI (so 2 năm của chính symbol), KHÔNG phải rủi ro tuyệt đối.
5. Mù với ĐUÔI thật (winsor + survivorship) → không thấy black-swan/chợ sập.
6. Own-symbol → bỏ tín hiệu hệ thống (cả thị trường cùng bão).
7. Nhịp tháng = chậm, bão trong-tháng vô hình.

**BÁO ĐỘNG (kết quả = dừng soi gấp), ngưỡng cắm TRƯỚC:**
- ⚠️ **#1: DL test vượt xa ~58–60%** (trần đã đo: persistence ~55%, signal +14%). Vượt = **LEAK/BUG**, KHÔNG phải thắng → dừng, săn rò rỉ. Kỳ vọng ĐÚNG = DL ≈ persistence.
- Over-confident (nói 90% đúng 60%) → soi calibration.
- Gap train≫test → overfit.
- Held-out SYMBOL dễ hơn held-out TIME → nghi leak normalize/split.
- Một symbol/lớp cao bất thường → nghi artifact data riêng.
- Thêm feature/capacity làm accuracy NHẢY (trái feature lab) → nghi leak feature mới.

**Nguyên tắc:** kỳ vọng DL ≈ 55–60%. **Trên trần = báo động, không phải mừng.** (Kỷ luật §4: biết trần trước, dừng ở trần, nghi mọi thứ vượt nó.)

## 16. BƯỚC 4 — PLAN DL (duyệt Beat 42): bung FULL toolkit CÓ KỶ LUẬT
**"Optimal" = kết quả HONEST tốt nhất trên VALIDATION rồi báo cáo TEST niêm phong — KHÔNG phải "số cao nhất" (= snoop).** 7 mẩu (làm từng mẩu):
1. **Dataset + SPLIT + baselines:** X=[current z, lag1] → band t+1; split TRAIN/VAL/**TEST-niêm-phong** (13 sym) + **4 sym held-out** (US2000/NZDUSD/ETHUSD/UKOIL); baselines majority/persistence/logistic = **mốc trần**.
2. **≥2 kiến trúc:** MLP nhỏ vs to/sâu; + input GIÀU (chuỗi K tháng vol-z) để test "cho nhiều hơn có giúp không" (kỳ vọng: không).
3. **≥2 optimizer:** GD/momentum/Adam + convergence (loss/epoch).
4. **Hyperparameter search:** khai báo budget · chọn trên VALIDATION · mở test 1 lần.
5. **Regularization:** L2/dropout/early-stop.
6. **Derivations pure-numpy** (forward · CE-grad · backprop · optimizer update) — ăn điểm + tự implement.
7. **Đánh giá (Bước 5):** test + held-out-symbol · calibration · transfer; áp ALARMS §15.
**Kỷ luật:** chọn trên val KHÔNG phải test; config vượt trần ~60% = BÁO ĐỘNG; kỳ vọng DL ≈ persistence. **GPU không cần** (data nhỏ, numpy CPU). DL notebook riêng: `notebooks/weather_regime_dl.ipynb`.
