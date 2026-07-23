document.getElementById("salaryForm").addEventListener("submit", function (event) {

    let age = parseInt(document.getElementById("age").value);

    let experience = parseFloat(document.getElementById("experience").value);

    if (age < 18) {
        alert("Age must be at least 18 years.");
        event.preventDefault();
        return;
    }

    if (experience < 0) {
        alert("Experience cannot be negative.");
        event.preventDefault();
        return;
    }

    if (experience > age - 18) {
        alert("Experience cannot be greater than your working age.");
        event.preventDefault();
        return;
    }

});
