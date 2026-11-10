# AutoML Studio — Roadmap / خارطة الطريق

> Learning/engineering project, not financial advice. / مشروع تعليمي وهندسي وليس نصيحة مالية.
>
> Legend / المفتاح: ✅ done / تم · 🔲 not done / لم يتم · 💡 idea, not committed / فكرة غير ملتزَم بها

---

## 1. Done / ما تم إنجازه ✅

### Core AutoML / أساس الـ AutoML

| ✅ | English | العربية |
|---|---|---|
| ✅ | Task detection (classification / regression) with manual override | كشف نوع المهمة (تصنيف / انحدار) مع إمكانية التغيير يدوياً |
| ✅ | Cleaning: duplicates, missing target, inf, constant / ID-like / mostly-empty columns, numeric text | التنظيف: المكرر، الهدف المفقود، inf، الأعمدة الثابتة أو الشبيهة بالـ ID أو الفارغة غالباً، النص الرقمي |
| ✅ | Preprocessing inside one Pipeline: imputation, outlier clipping (IQR), scaling, one-hot, feature selection | المعالجة داخل Pipeline واحد: تعويض المفقود، قص الشواذ (IQR)، التقييس، الترميز، اختيار الميزات |
| ✅ | Date/time columns expanded to hour / weekday / day / month | أعمدة التاريخ تتحول إلى ساعة / يوم أسبوع / يوم / شهر |
| ✅ | Models compared by cross-validation; champion chosen | مقارنة النماذج بالتحقق المتقاطع واختيار البطل |
| ✅ | Optional XGBoost and LightGBM candidates | XGBoost وLightGBM كمرشحين اختياريين |
| ✅ | Optional hyperparameter tuning (adopted only if CV improves) | ضبط اختياري للإعدادات (يُعتمد فقط إن تحسن الـ CV) |
| ✅ | Export whole pipeline as one `.pkl` + leaderboard `.csv` + model card `.md` | تصدير الـ Pipeline كاملاً في `.pkl` واحد مع leaderboard وmodel card |

### Honest evaluation / تقييم صادق

| ✅ | English | العربية |
|---|---|---|
| ✅ | Chronological split + purge gap (no leakage), also between CV folds | تقسيم زمني مع فجوة (purge gap) لمنع التسرب، وبين طيّات الـ CV أيضاً |
| ✅ | Baseline shown next to every metric | الـ baseline يُعرض بجانب كل مقياس |
| ✅ | Permutation importance on unseen test data | أهمية الميزات (permutation) على بيانات الاختبار |
| ✅ | Directional accuracy and fee-aware long/flat backtest vs buy-and-hold | دقة الاتجاه وbacktest مع الرسوم مقابل الشراء والاحتفاظ |
| ✅ | Friendly errors and warnings for tiny datasets (CV folds reduced automatically) | أخطاء وتحذيرات واضحة للبيانات القليلة (تقليل الطيّات تلقائياً) |
| ✅ | 33 automated tests, including a test that features use only past data | 33 اختباراً آلياً، منها اختبار أن الميزات تستخدم الماضي فقط |

### App and market / التطبيق والسوق

| ✅ | English | العربية |
|---|---|---|
| ✅ | Streamlit app: Live Market, Model Dashboard, Predict | تطبيق Streamlit: Live Market وModel Dashboard وPredict |
| ✅ | Live XRP/USDT candles from the public Binance API (no key) | شموع XRP/USDT حيّة من Binance العامة (بدون مفتاح) |
| ✅ | 12 market features (returns, MA ratios, volatility, RSI, candle shape, volume change) | 12 ميزة سوقية (عوائد، نسب المتوسطات، تباين، RSI، شكل الشمعة، تغير الحجم) |
| ✅ | Predict page: session champion or uploaded `.pkl`; live candles or CSV | صفحة Predict: بطل الجلسة أو ملف `.pkl`؛ شموع حيّة أو CSV |
| ✅ | Colored class badge, probability chart, interval-mismatch warning | شارة ملونة للفئة، مخطط احتمالات، تحذير عدم تطابق الـ interval |
| ✅ | Docs: README, HANDOFF, diagrams, Arabic video guide | الوثائق: README وHANDOFF ومخططات ودليل الفيديو بالعربية |

**Result to state honestly / النتيجة التي نقولها بصدق:** the models show no predictive edge on XRP/USDT (R² ≈ −0.06 vs baseline ≈ −0.047; directional accuracy 52–53% vs 55% majority). The value of the project is a correct, leak-free method. / النماذج لم تُظهر ميزة تنبؤية على XRP/USDT، وقيمة المشروع في المنهجية السليمة الخالية من التسرب.

---

## 2. Remaining / المتبقي 🔲

| Priority / الأولوية | English | العربية |
|---|---|---|
| High / عالية | More market features (MACD, Bollinger bands, hour of day) and other symbols | ميزات سوقية أكثر (MACD وBollinger وساعة اليوم) ورموز عملات أخرى |
| High / عالية | Optional MLflow run tracking | تتبع التجارب بـ MLflow (اختياري) |
| Medium / متوسطة | Block or confirm predictions when the live interval differs from training | منع التنبؤ أو طلب تأكيد عند اختلاف الـ interval عن التدريب |
| Medium / متوسطة | Class-distribution chart for classification (how many rows per class) | مخطط توزيع الفئات في التصنيف (عدد الصفوف لكل فئة) |
| Medium / متوسطة | Auto-refresh live predictions | تحديث تلقائي للتنبؤات الحيّة |
| Low / منخفضة | Refresh `HANDOFF.md` (it still says the UI was never run) | تحديث `HANDOFF.md` (ما زال يقول إن الواجهة لم تُشغَّل) |
| Pending / معلّق | "Modification 2" and "Modification 3": text never received from the user | «التعديل 2 و3»: لم يصل نصهما من المستخدم بعد |

---

## 3. Possible future development / تطوير محتمل مستقبلاً 💡

Not committed; each item needs its own design and testing. / غير ملتزَم بها؛ كل بند يحتاج تصميماً واختباراً مستقلاً.

- 💡 **Walk-forward evaluation** over many periods instead of one test block, for a steadier estimate. / تقييم walk-forward على فترات متعددة بدل كتلة اختبار واحدة لتقدير أثبت.
- 💡 **Probability calibration** and a confidence threshold, so weak signals show as NEUTRAL. / معايرة الاحتمالات وعتبة ثقة حتى تظهر الإشارات الضعيفة NEUTRAL.
- 💡 **Richer backtest:** slippage, position sizing, stop-loss. / backtest أغنى: انزلاق السعر وحجم الصفقة ووقف الخسارة.
- 💡 **Extra data:** order book, funding rates, other coins' prices (e.g. BTC) as features. / بيانات إضافية: دفتر الأوامر ومعدلات التمويل وأسعار عملات أخرى كالبتكوين.
- 💡 **Lag and rolling features for uploaded CSVs** (today only calendar parts). / ميزات lag وrolling لملفات CSV المرفوعة (اليوم أجزاء التاريخ فقط).
- 💡 **Per-column outlier control** and early stopping in tuning. / تحكم بالشواذ لكل عمود وإيقاف مبكر في الضبط.
- 💡 **Safer model file format** (e.g. ONNX) so loading does not run pickle code. / صيغة نموذج أكثر أماناً (مثل ONNX) حتى لا يشغّل التحميل كود pickle.
- 💡 **Packaging:** Docker image, CI running `pytest` on every change, hosted deployment. / التغليف: صورة Docker وCI يشغّل الاختبارات عند كل تغيير ونشر على استضافة.
- 💡 **Browser-level UI tests** for file upload paths (today covered at engine level). / اختبارات واجهة في المتصفح لمسارات رفع الملفات (اليوم على مستوى المحرك).

---

## 4. Requirements files / ملفات المتطلبات

| File / الملف | Purpose / الغرض | Install / التثبيت |
|---|---|---|
| `requirements.txt` | Core: everything the app needs / الأساسي: كل ما يحتاجه التطبيق | `pip install -r requirements.txt` |
| `requirements-extra.txt` | Optional XGBoost (CPU) + LightGBM / نماذج اختيارية | `pip install -r requirements-extra.txt` |
| `requirements-dev.txt` | Core + pytest / الأساسي مع pytest | `pip install -r requirements-dev.txt` |

Python 3.10+. Tested with pandas 3.0, scikit-learn 1.9, Streamlit 1.66.
