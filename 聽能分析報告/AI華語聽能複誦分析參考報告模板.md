# AI 華語聽能複誦分析參考報告模板

## 一、模板定位

本報告模板定位為「AI 自動分析產生之聽能複誦參考報告」，用於整理受試者在華語聽能複誦任務中的表現、主要聽辨混淆型態，以及後續聽能訓練建議。

本報告不應被解釋為臨床診斷報告，也不應直接判定患者存在構音或發音障礙。報告中的聲母、韻母、聲調、發音位置與發音方法，主要用於描述「聽辨混淆型態」，而非判斷患者是否發音錯誤。

## 二、報告模板

```text
AI 華語聽能複誦分析參考報告

一、測驗資訊
受試者編號：
測驗日期：
刺激來源：Qwen3-TTS
測驗材料：單音節詞 / 雙音節詞 / 短句
測驗條件：安靜 / 噪音
SNR：
題數：
播放音量：
輸入方式：口語複誦
分析模型：ASR + 聲調辨識模型

二、整體表現摘要
整體複誦正確率：
關鍵字正確率：
聲母聽辨正確率：
韻母聽辨正確率：
聲調聽辨正確率：
ASR 可判讀比例：
低信心題數：

三、主要聽辨混淆型態

1. 聲調混淆
   - 最常見混淆：
   - 聲調錯誤率：
   - 代表題目：

2. 聲母聽辨混淆
   - 送氣 / 不送氣混淆：
   - 擦音 / 塞擦音混淆：
   - 捲舌音 / 舌面音混淆：
   - 高頻聲母混淆：

3. 韻母聽辨混淆
   - 單母音混淆：
   - 複合韻母混淆：
   - 鼻音韻尾 -n / -ng 混淆：

4. 詞句層級表現
   - 單音節詞正確率：
   - 雙音節詞正確率：
   - 短句關鍵字正確率：
   - 噪音條件下錯誤增加項目：

四、AI 判讀信心與人工複核建議
以下情況建議人工複核：
- 錄音品質不佳
- ASR 信心低
- ASR 與聲調模型結果不一致
- 患者回答含糊或多次自我修正
- 錯誤可能來自口語表達而非聽辨

五、聽能訓練建議
根據本次錯誤型態，建議優先加強：
1. 聲調最小對比聽辨：
2. 聲母聽辨：
3. 韻母聽辨：
4. 噪音中關鍵字辨識：
5. 多說話者語音辨識：

六、備註
本報告為 AI 自動分析產生之參考結果，反映受試者在本次複誦任務中的語音聽辨表現。結果可能受注意力、短期記憶、口語輸出、錄音品質與 ASR 辨識誤差影響，需由專業人員結合臨床資料判讀。
```

## 三、欄位說明

### 1. 測驗資訊

此區塊記錄測驗條件，例如刺激來源、測驗材料、噪音條件、SNR、題數與模型版本。這些條件會影響聽辨結果，因此應保留於報告中，方便後續比較與研究分析。

### 2. 整體表現摘要

此區塊提供量化分數，包括整體複誦正確率、關鍵字正確率、聲母聽辨正確率、韻母聽辨正確率與聲調聽辨正確率。這些分數可用於和聽力師人工評分進行一致性驗證。

### 3. 主要聽辨混淆型態

此區塊為報告核心。聲母、韻母、聲調與發音位置等資訊不應被描述為患者的發音缺陷，而應用來描述患者在聽覺辨識時容易混淆的語音對比。

例如：

- 「二聲與三聲混淆」應描述為聲調聽辨困難。
- 「zh/ch/sh 與 j/q/x 混淆」應描述為捲舌音與舌面音之聽辨混淆。
- 「-n 與 -ng 混淆」應描述為鼻音韻尾聽辨混淆。

### 4. AI 判讀信心與人工複核建議

此區塊用於標記 AI 分析結果可能不穩定的案例。若錄音品質不佳、ASR 信心低、ASR 與聲調模型結果不一致，或患者回答含糊，應標記為需要人工複核。

### 5. 聽能訓練建議

此區塊根據錯誤型態提供聽能訓練方向。建議內容應使用「聽辨訓練」語言，例如「聲調最小對比聽辨」、「鼻音韻尾聽辨」、「噪音中關鍵字辨識」，避免寫成構音治療或發音矯正。

## 四、建議用語

### 建議使用

- 聲母聽辨正確率
- 韻母聽辨正確率
- 聲調聽辨正確率
- 聽辨混淆型態
- 聲學 / 音韻對比
- 建議加強聽辨訓練
- 需人工複核
- AI 參考分析

### 避免使用

- 發音錯誤診斷
- 構音障礙判定
- AI 診斷結果
- 患者發音位置錯誤
- 患者需要構音治療
- AI 取代聽力師判讀

## 五、研究驗證建議

後續研究可將本報告模板作為 AI 自動評分輸出格式，並進行以下驗證：

1. 比較 AI 自動評分與聽力師人工評分之一致性。
2. 比較 AI 聽辨混淆分類與聽力師錯誤分析之一致性。
3. 分析不同錯誤類型中，AI 最容易誤判的項目。
4. 評估聽力師對 AI 參考報告的可讀性、臨床可用性與信任程度。

可使用的統計指標包括：

- Accuracy
- Word error rate
- Character error rate
- Tone accuracy
- Cohen's kappa
- Intraclass correlation coefficient
- Bland-Altman analysis

## 六、文獻依據

1. Winn 等人指出，語音辨識分數不等於完整的聽覺努力表現，錯誤型態也會影響聽覺處理負荷。因此，報告中加入錯誤型態分析是合理的。  
   Winn, M. B. et al. Listening Effort Is Not the Same as Speech Intelligibility Score. Trends in Hearing, 2020.

2. 華語聲調是詞義辨識的重要線索，聽損與人工耳蝸使用者可能在聲調與 F0 線索上出現聽辨困難。因此，報告中納入聲調聽辨正確率與聲調混淆型態是合理的。  
   Chen, Y. et al. The Role of Lexical Tone Information in the Recognition of Mandarin Sentences in Listeners With Hearing Aids. Ear and Hearing, 2019.  
   Chang, Y. P. et al. Mandarin Tone and Vowel Recognition in Cochlear Implant Users: Effects of Talker Variability and Bimodal Hearing. Ear and Hearing, 2016.

3. 個人化或適性聽能訓練可依據患者能力與錯誤表現調整訓練內容。因此，根據錯誤型態產生聽能訓練建議具有研究依據。  
   Gnadlinger, F. et al. Incorporating an Intelligent Tutoring System Into a Game-Based Auditory Rehabilitation Training for Adult Cochlear Implant Recipients. JMIR Serious Games, 2024.  
   Dornhoffer, J. R. et al. Systematic Review of Auditory Training Outcomes in Adult Cochlear Implant Recipients and Meta-Analysis of Outcomes. Journal of Clinical Medicine, 2024.

4. AI 或 ASR 輔助評分應與人工評分比較，並以 ICC、Bland-Altman、WER 等方法驗證一致性與可靠度。  
   Zhang, V. W. et al. Automated Speech Intelligibility Assessment Using AI-Based Transcription in Children with Cochlear Implants, Hearing Aids, and Normal Hearing. Journal of Clinical Medicine, 2025.

