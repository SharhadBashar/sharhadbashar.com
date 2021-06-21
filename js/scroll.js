$(function() {
  $('a[href*=#]').on('click', function(e) {
    e.preventDefault();
    $('html, body').animate({ scrollTop: $($(this).attr('href')).offset().top}, 500, 'linear');
  });
});

var quoteNumber;
function newQuote() {
  var quotes = [
   "\"Dude, suckin' at something is the first step at being sorta good at something\"<br>-  Jake <small><em>(Adventure Time)</em></small>",
   "\"Either I will find a way, or I will make one\"<br> - Philip Sidney",
   "\"Our greatest weakness lies in giving up. The most certain way to succeed is always to try just one more time\"<br>- Thomas A. Edison",
   "\"You are never too old to set another goal or to dream a new dream\"<br>- C.S Lewis",
   "\"If you can dream it, you can do it\"<br>- Walt Disney",
   "\"Never give up, for that is just the place and time that the tide will turn\"<br>- Harriet Beecher Stowe",
   "\"I know where I'm going and I know the truth, and I don't have to be what you want me to be. I'm free to be what I want\"<br>- Muhammad Ali",
   "\"If you always put limit on everything you do, physical or anything else. It will spread into your work and into your life. There are no limits. There are only plateaus, and you must not stay there, you must go beyond them\"<br>- Bruce Lee",
   "\"Imagination is more important that knowledge\"<br>- Albert Einstein",
   "\"The people who are crazy enough to think they can change the world are the ones who do\" <br/>- Steve Jobs",
   "\"My greatest strength is my love for my people, my greatest weakness is that I love them too much\" <br>- Sheikh Mujibur Rahman"
  ]
  var randonNumber = Math.floor(Math.random() * (quotes.length));
  quoteNumber = randonNumber
  return quotes[randonNumber];
}

document.getElementById('famousQuote').innerHTML = newQuote();

function image(){
  var picture = [
    "../Images/jake.png",
    "../Images/philip.jpg",
    "../Images/thomas.jpg",
    "../Images/lewis.jpg",
    "../Images/disney.jpg",
    "../Images/harriet.jpg",
    "../Images/ali.jpg",
    "../Images/lee.jpg",
    "../Images/albert.jpg",
    "../Images/jobs.jpg",
    "../Images/mujibur.jpg"
  ]
  return picture[quoteNumber];
}

document.getElementById('famousImage').src = image();






