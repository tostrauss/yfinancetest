## 1. Options Fundamentals

Options are versatile financial instruments that provide an investor like yo
with flexibility to speculate, hedge risk, or generate income. Two of the most common types are **Call Options** and **Put Options**.

### Call Options

- **Definition:**  
  A call option gives the buyer the **right, but not the obligation**, to purchase the underlying asset (e.g., a stock) at a predetermined **strike price** on or before a specified **expiration date**.

- **Key Characteristics:**
  - **Premium:** The price paid by the buyer to the seller (writer) for the option.
  - **American vs. European:** American options can be exercised any time before expiration; European options can only be exercised on the expiration date.
  - **Leverage:** Allows investors to control more shares with less capital.
  
- **Example:**  
  Suppose you purchase a call option for stock XYZ with:
  - **Strike Price:** $100
  - **Premium:** $5 per share
  - **Expiration:** 1 month from now
  
  If, before expiration, XYZ rises to $120:
  - **Intrinsic Value:** $120 (current price) - $100 (strike price) = $20 per share.
  - **Profit (ignoring transaction costs):** $20 (intrinsic value) - $5 (premium) = $15 per share.
  
  In this case, the option provides a leveraged gain compared to buying the stock outright.

### Put Options

- **Definition:**  
  A put option gives the buyer the **right, but not the obligation**, to sell the underlying asset at a predetermined **strike price** on or before a specified **expiration date**.

- **Key Characteristics:**
  - **Premium:** The cost to purchase the put option.
  - **Downside Protection:** Can be used as a hedge against a decline in the price of the underlying asset.
  
- **Example:**  
  Suppose you purchase a put option for stock XYZ with:
  - **Strike Price:** $100
  - **Premium:** $4 per share
  - **Expiration:** 1 month from now
  
  If, before expiration, XYZ falls to $80:
  - **Intrinsic Value:** $100 (strike price) - $80 (current price) = $20 per share.
  - **Profit (ignoring transaction costs):** $20 (intrinsic value) - $4 (premium) = $16 per share.
  
  This put option acts as insurance, allowing you to limit losses if the stock price declines.

## 2. Hedging with Options

**Hedging** is a risk management strategy used to offset potential losses in an investment. Options are popular hedging instruments because they provide insurance-like protection without requiring the full capital commitment of the underlying asset.

### Protective Put

- **Strategy:**  
  Buy put options while holding the underlying shares.
  
- **Purpose:**  
  Protects against a decline in the stock’s price.
  
- **Example:**  
  You own 100 shares of Company ABC at $50 per share. To protect against a significant drop, you purchase one put option (each option typically covers 100 shares) with a strike price of $50 for a premium of $2 per share.  
  - **Outcome:** If ABC falls to $40, the increase in the put option’s value offsets the loss on the shares, limiting your downside risk.

### Covered Call

- **Strategy:**  
  Hold shares of the underlying asset and sell call options against those shares.
  
- **Purpose:**  
  Generate extra income (the premium) on the held shares while potentially capping the upside if the stock exceeds the strike price.
  
- **Example:**  
  You own shares of Company ABC at $50 per share and sell a call option with a strike price of $55 for a premium of $3 per share.
  - **Outcome:**  
    - If the stock remains below $55, you keep the premium and the shares.  
    - If the stock rises above $55, you may have to sell your shares at $55, capping your profit, but still keeping the premium as additional income.

### Collar Strategy

- **Strategy:**  
  Combine buying a protective put and selling a covered call on the same underlying asset.
  
- **Purpose:**  
  Limits both upside potential and downside risk, creating a “collar” around the stock’s value.
  
- **Example:**  
  Holding shares of XYZ, you might:
  - **Buy a put** with a strike price at $50.
  - **Sell a call** with a strike price at $60.
  
  This strategy reduces the net cost of hedging, though it also limits your gains if the stock surges past $60.

---

## 3. Options Strategies

Advanced options strategies are tailored to various market expectations—from low volatility to directional bets. Below are some popular strategies with examples.

### Butterfly Spread

- **Overview:**  
  A limited-risk, limited-reward strategy that involves multiple options with three strike prices.
  
- **How It Works:**  
  - Buy 1 call at a lower strike (e.g., $90).
  - Sell 2 calls at a middle strike (e.g., $100).
  - Buy 1 call at a higher strike (e.g., $110).
  
- **Market Outlook:**  
  Profits are maximized when the underlying asset is at or near the middle strike price at expiration.
  
- **Risk/Reward:**  
  Both maximum loss and maximum gain are capped. This strategy benefits from low volatility.

### Condor Spread

- **Overview:**  
  Similar to the butterfly spread but uses four different strike prices to create a wider profit zone.
  
- **How It Works:**  
  - Buy a call at strike $90.
  - Sell a call at strike $100.
  - Sell a call at strike $110.
  - Buy a call at strike $120.
  
- **Market Outlook:**  
  Best used when the underlying asset is expected to have low volatility and remain within a specific range.
  
- **Risk/Reward:**  
  Limited risk and reward; it provides a smoother profit curve compared to the butterfly.

### Bull and Bear Spreads

- **Bull Spread (Call Spread):**  
  - **Strategy:** Buy a call at a lower strike and sell a call at a higher strike.
  - **Market Outlook:** Expect a moderate rise in the asset’s price.
  - **Example:** Stock XYZ is trading at $100. Buy a call at $100 and sell a call at $110. Maximum profit is achieved if the stock is above $110 at expiration, while losses are limited to the net premium paid.

- **Bear Spread (Put Spread):**  
  - **Strategy:** Buy a put at a higher strike and sell a put at a lower strike.
  - **Market Outlook:** Expect a moderate decline in the asset’s price.
  - **Example:** Stock XYZ is at $100. Buy a put at $100 and sell a put at $90. Maximum profit is realized if the stock falls below $90, with risk limited to the net premium paid.

### “Free Lunch” Strategies

- **Overview:**  
  Sometimes known as synthetic positions or risk reversals, these strategies aim to create a net-zero or low-cost position by offsetting premiums.
  
- **Example (Risk Reversal):**  
  - Sell a put option while simultaneously buying a call option.
  - This can create a synthetic long position on the underlying asset with little to no net premium cost.
  
- **Important Note:**  
  The term “free lunch” is a misnomer—while such strategies may reduce upfront costs, they carry risks that must be understood and managed.

---
## 4. Financial Ratios and Metrics

Financial ratios help assess a company’s operational efficiency, liquidity, and financial stability. They are crucial for both investors and analysts.

1. Current Ratio
+ Description: The current ratio is a liquidity ratio that measures a company´s ability to cover its short-term obligations with its short-term assets.
+ Formula: Current Ratio = Current Assets / Current Liabilities
+ Interpretation: A ratio above 1 indicates that the company has more assets than liabilities due within the next year. A low current ratio might suggest liquidity problems, while a very high ratio could indicate excess inventory or inefficient use of cash.

2. Debt to Equity Ratio
+ Description: THis ratio assesses a company´s financial leverage by comparing its total liabilities to its shareholders´ equity.
+ Formula: Debt to Equity Ratio = Total Liabilities / Shareholders´ Equity. 
+ Interpretation: A higher ratio might indicate that the company is heavily financed by debt, which could be risky. A lower ratio suggests a healthier balance between debt and equity financing.

3. Return on Equity (ROE)
+ Description: ROE measures how effectively a company is using it´s equity to generate profit.
+ Formula: ROE = Net Income / Shareholder´s Equity
+ Interpretation: A higher ROE indicates more effective use of equity to generate profits. However, a very high ROE could also indicate excessive financial leverage. 

4. Gross Profit Margin
+ Description: This ratio indicates the percentage of revenue that exceeds the cost of goods sold (COGS). It reflects the efficiency of producting as well as pricing.
+ Formula: Gross Profit Margin = (Revenue - COGS) / Revenue
+ Interpretation: A higher gross profit margin suggests that the company is selling goods at a higher markup or producing them at a lower cost. A lower margin might indicate pricing pressure, higher production cost, or ineffiencies.

5. Net Profit Margin
+ Description: The net profit marging reflects the percentage of revenue left after all expenses have been deducted. It shows how effectively a company is managing it´s costs and operations.
+ Formula: Net Profit Margin = Net Income / Revenue
+ Interpretation: A higher net profit margin indicates a more profitable company, while a lower margin might suggest high expenses, poor pricing strategy, or operational ineffiencies.


###
These ratios provide valuable insights when analyzed over time and compared to industry benchmarks or competitors. They help stakeholders make informed decisions regarding investments, lending, and the componay´s overall finaincial strategy.
###


Return on Assets (ROA)
* Description: ROA measures the amount of profit a company earns for every dollar of its assets. It indicates how efficiently a company can convert its assets into net income.
* Formula: ROA = Net Income / Average Total Assets
* Average Total Assets can be calculated as (Beginning Total Assets + Ending Total Assets)/2 for a specific period.
* Interpretation: A higher ROA indicates more efficient se of assets to generate profit. A ROA that is higher than industry average suggests competitive advantage in asset utilization. A lower ROA may indicate inefficiency or mismanagement of assets. 
* It is important to compare ROA among companies within the same industry, as diferent industires have varying capital intensities and operational structures. ROA provides a holistic view of a company´s operational efficiency and its particular relevant for investors and analysts looking to evaluate a company´s performance relative to its asset base.

Cash Flow Ratio 
* Description: The Cash Flow Ratio measures a company´s ability to cover its short-term liabilities with its operating cash flows, giving an indication of the company´s liquidity and efficiency in using its cash.
* Formula: Cash Flow Ratio = Operating Cash Flow / Current Liabilites
* Interpretation: A ratio greater than 1 indicates that the company generates enough cash from its operations to cover its short-term liabilities, suggesting good liquidity. A ratio less than 1 may indicate potential liquidity problems, implying that the company might sturggle to pay off its short-term obligations if they all came due at once. Like other ratios, the Cash Flow Ratio should be compared to industry benchmarks and historical company performance for a more comprehensive analysis.
* This ratio is particularly useful for stakeholders to assess how well a company can sustain its operations and meet its obligations using the cash generated from its core business activities, tahter than relying on external financinf or liquidating assets.


#####
- An RSI above 50 indicate "hold".
- An RSI leaving 70 going negative indicates "sell".
- An RSI leaving 30 going positive indicates "buy".
- An RSI moving opposite of Stock-Price indicates potential reversal. 


