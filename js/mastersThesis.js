function password() {
	var password = prompt("Please enter the password");
	if (password === "sharhadthesis") {
		document.getElementById('download').click();
	}
	else {
		alert("Password incorrect");
	}
}