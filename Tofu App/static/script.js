// Define slides data
const slidesData = {
    whatIsAStock: [ 
        { title: "1/8", content: "The first step in trading stocks is knowing what a stock is. Stocks, also known as equities or shares, are slices of a company that denote partial ownership of the company." },
        { title: "2/8", content: "Companies sell stocks to raise money. Depending on the type of company and the kind of stock, stockholders may receive dividends (a distribution of company earnings) and the right to vote at shareholder meetings." },
        { title: "3/8", content: "There are two types of shares: common and preferred. Common stock (or common shares) generally offer higher rewards but also carry greater risk, especially when investing in smaller companies."},
        { title: "4/8", content: "Owners of common shares are entitled to vote at shareholder meetings, with the strength of their vote changing proportionally with the number of shares they own."},
        { title: "5/8", content: "Common shares are also directly tied to the market and the value of them can fluctuate immensely. These shares are the most commonly traded on the stock market and are what most people are talking about when they mention stocks."},
        { title: "6/8", content: "Although common shares can receive dividends, they are not guaranteed. This is why some investors like preferred shares. Although these shares don't carry the power to vote in shareholder meetings, they are more stable than common shares."},
        { title: "7/8", content: "Preferred shares give shareholders a fixed dividend rate, meaning that no matter how well or poorly the company does, owners of preferred shares will receive a fixed payout either quarterly, semi-annually, or annually depending on the company."},
        { title: "8/8", content: "Preferred shares also limit downside for investors during catastrophic failures of a company such as bankruptcy. Before any common stock holders are paid out, a company must first pay out all preferred stock holders."}
    ],
    valueOfMoney: [
        { title: "Value of money", content: "Insert lesson on value of money." },
        { title: "Pythagorean Theorem", content: "Understanding the Pythagorean theorem." }
    ],
    pairsTrading: [
        { title: "1/9", content: "This lesson we will cover a strategy known as pairs trading, sometimes referred to as statistical arbitrage; it's a market neutral strategy where a trader can bet on a price convergence or divergence of two related equities." },
        { title: "2/9", content: "By using fundamental and technical analysis, traders can identify two or more equities with a high convergence level, such as Pepsi (ticker PEP) and Coke (ticker KO). From here, more analysis can be done to decide whether the trader wants to bet on convergence or divergence."},
        { title: "3/9", content: "In the Coke and Pepsi example, a trader can go long Coke and short Pepsi to bet on those equities price convergence, whereas to bet on the stock prices diverging, they would go long Pepsi and short Coke. (Pepsi currently trades higher per share than Coke)"},
        { title: "4/9", content: "By making a short and a long trade, a trader is protected if both equities fall in price or rise in price. Because the only way a loss is incurred is if the stock prices diverge after placing a convergent bet or vice versa, the only way a trader profits is the opposite."},
        { title: "5/9", content: "Pairs trading is a viable strategy that requires a lot of funamental and technical analysis and is useful in protecting a trader from market instability."},
        { title: "6/9", content: "Though the terms pairs trading and statistical arbitrage are often used interchangably, there are some core differences. While pairs trading is relatively simple and only requires basic statistical tools, most statistical arbitrage strategies take advantage of high frequency trading algorithms."},
        { title: "7/9", content: "Statistical arbitrage strategies and pairs trading both rely on market prices diverging or converging on a predicted normal, known as a mean reversion (prices converge to the mean) or an overextension (the market has made an extended divergence)."},
        { title: "8/9", content: "One thing pairs traders should take into account is that in most scenarios, they are competing with hedgefunds and high frequency traders that can execute trades in a fraction of a second by using algorithms that require an immense amount of statistical analysis."},
        { title: "9/9", content: "Because of this, many traders consider pairs trading to be a more theoretical strategy than one that can be effectively implemented. However the concepts of pairs trading can be utilized in other trading strategies to hedge or protect a trader from large market swings."}
    ],
    chemistry: [
        { title: "Periodic Table", content: "Introduction to the periodic table." },
        { title: "Chemical Reactions", content: "Basics of chemical reactions." }
    ],
    bollingerBands: [
        { title: "1/8", content: "As you progress in your trading journey, you may want to know what signals skilled traders use to help them identify good trades"},
        { title: "2/8", content: "Bollinger bands are one of those signals, which are commonly referred to as indicators. Bollinger bands were created by John Bollinger in the 1980s as a tool to help traders evaluate the movement of an asset's price over time as well as its volatility."},
        { title: "3/8", content: "A Bollinger band consists of three bands, the middle of which is a moving average, as well as an upper and lower band. The upper and lower bands are calculated by adding or subtracting a specified number of standard deviations (usually 2) to the middle band"},
        { title: "4/8", content: "The moving average these bands reference is most commonly 20 periods, though this can be changed, and is used to help a trader identify strength and direction of trends, overbought or oversold conditions, and gives a good indicator of an asset's volatility."},
        { 
            title: "5/8",
            content: "Here is an example of what bollinger bands overlayed on a stock chart could look like",
            image: imageBasePath + "bollingerbands.jpeg"
        },
        { title: "6/8", content: "Prices near the upper band may indicate overbought conditions, which could be a good signal to sell, whereas prices near the lower band may indicate oversold conditions and could be a good signal to buy. The distance between the upper and lower bands also reflects market volatility"},
        { title: "7/8", content: "One thing traders who utilize Bollinger bands look out for is what's called a Bollinger squeeze. This is when the upper and lower bands contract significantly, indicating low volatility and can signal a potential breakout one way or another."},
        { title: "8/8", content: "Though Bollinger bands may help to provide insight into how a stock could move, it is important to take other factors into account before rushing into a trade as they have their limitations and can sometimes produce false signals, especially in volatile markets."}
    ]
};


// Toggle visibility of lessons
function toggleLessons(category) {
    event.stopPropagation(); // Added to prevent event bubbling
    const lessons = document.getElementById(`${category}-lessons`);
    lessons.style.display = lessons.style.display === 'block' ? 'none' : 'block';
}


// Show slideshow for a specific lesson
function showSlideshow(lesson) {
    event.stopPropagation(); // Added to prevent event bubbling
    const slideshow = document.getElementById('slideshow');
    
    // Set initial slide
    currentLesson = lesson;
    currentSlideIndex = 0;
    updateSlide();

    // Show slideshow
    slideshow.style.display = 'block';

    // debugging
    console.log("Showing slideshow for:", lesson);
    console.log("Available lessons:", Object.keys(slidesData));
}


// Slideshow functionality
let currentLesson = null;
let currentSlideIndex = 0;

function updateSlide() {
    const slideTitle = document.getElementById('slide-title');
    const slideContent = document.getElementById('slide-content');
    const slideImage = document.getElementById('slide-image');

    if (currentLesson && slidesData[currentLesson]) {
        const slide = slidesData[currentLesson][currentSlideIndex];
        slideTitle.textContent = slide.title;
        slideContent.textContent = slide.content;

        // Update image if it exists
        if (slide.image) {
            slideImage.src = slide.image;
            slideImage.style.display = 'block';
            console.log("Loading image:", slide.image);
        } else {
            slideImage.style.display = 'none';
        }
    }

    //debugging
    console.log("Updating slide for lesson:", currentLesson, "index:", currentSlideIndex);
    if (currentLesson && slidesData[currentLesson]) {
        console.log("Slide data:", slidesData[currentLesson][currentSlideIndex]);
    } else {
        console.log("No slide data found!");
}
}

function nextSlide() {
    if (currentLesson && slidesData[currentLesson]) {
        currentSlideIndex = (currentSlideIndex + 1) % slidesData[currentLesson].length;
        updateSlide();
    }
}

function prevSlide() {
    if (currentLesson && slidesData[currentLesson]) {
        currentSlideIndex = (currentSlideIndex - 1 + slidesData[currentLesson].length) % slidesData[currentLesson].length;
        updateSlide();
    }
}