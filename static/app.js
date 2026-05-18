const users = [
  { id: "s12345", password: "1234", role: "student" },
  { id: "a98765", password: "1234", role: "advisor" },
  { id: "e11111", password: "1234", role: "employee" }
];

const form = document.getElementById("loginForm");
const errorMessage = document.getElementById("errorMessage");

if (form) {
  form.addEventListener("submit", function (event) {
    event.preventDefault();

    const userId = document.getElementById("userId").value.trim();
    const password = document.getElementById("password").value.trim();
    const role = document.getElementById("role").value;

    if (!role) {
      errorMessage.textContent = "الرجاء اختيار نوع المستخدم.";
      return;
    }

    if (!userId || !password) {
      errorMessage.textContent = "الرجاء تعبئة جميع الحقول.";
      return;
    }

    const user = users.find(function (u) {
      return u.id === userId && u.password === password && u.role === role;
    });

    if (!user) {
      errorMessage.textContent = "بيانات الدخول غير صحيحة.";
      return;
    }

    errorMessage.textContent = "";

    localStorage.setItem(
      "currentUser",
      JSON.stringify({ id: user.id, role: user.role })
    );

    if (user.role === "student") {
      window.location.href = "/student-page";
    } else if (user.role === "advisor") {
      window.location.href = "/advisor-page";
    } else if (user.role === "employee") {
      window.location.href = "/employee-page";
    }
  });
}

const guestLogin = document.getElementById("guestLogin");

if (guestLogin) {
  guestLogin.addEventListener("click", function () {
    localStorage.setItem(
      "currentUser",
      JSON.stringify({ id: "guest", role: "guest" })
    );

    window.location.href = "/student-page";
  });
}
