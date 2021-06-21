function newQuote() {
  var randonNumber = Math.floor(Math.random() * (quotes.lenght));
  document.getElementById('section_quote').innerHTML = quotes[randomNumber];
}

var quotes = [
 'hello',
 'buy',
 'kemon acho'
]