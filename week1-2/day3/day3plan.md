# Plan for Day 3: Eigenvectors & Eigenvalues

**Morning Session: Theory & Intuition (2.5 hours)**

*   **0:00 - 1:00 (60 mins): Core Concepts**
    *   **(50 mins):** Watch the definitive 3Blue1Brown video on the topic.
        *   *Eigenvectors and eigenvalues*
    *   **(10 mins):** Break. Let it sink in. The core idea is simple but profound: "What vectors stay on their own span during a transformation?"

*   **1:00 - 2:00 (60 mins): Active Recall & Deeper Understanding**
    *   **(50 mins):** In your notebook, write down answers from memory:
        *   In your own words, what is an eigenvector of a matrix? What's special about its direction?
        *   What does the corresponding eigenvalue represent? What if the eigenvalue is 2? What if it's 0.5? What if it's -1?
        *   Does a rotation matrix in 2D have any real eigenvectors? Why or why not? (Hint: what vector doesn't change direction when you rotate a piece of paper?)
        *   What is the geometric meaning of the equation `Av = λv`?
    *   Re-watch the video to refine your answers.
    *   **(10 mins):** Break.

*   **2:00 - 2:30 (30 mins): Formalize with Reading**
    *   **(30 mins):** Read the corresponding sections in *Linear Algebra Done Right*. Focus on the characteristic equation `det(A - λI) = 0`. Understand how this algebraic manipulation is a clever way to find the values of `λ` that make the matrix `A - λI` have a non-trivial null space (which is where the eigenvectors live).

---

**Mid-day Break (1 hour)**

---

**Afternoon Session: Practice & Implementation (4 hours)**

*   **3:30 - 5:00 (90 mins): Hands-on with NumPy**
    *   **(80 mins):** Open your `q1_foundations/linear_algebra.py` file.
        *   Use `np.linalg.eig()` to find the eigenvalues and eigenvectors of a few matrices.
        *   Start with a simple diagonal matrix, like `[[2, 0], [0, 3]]`. Can you predict the eigenvectors and eigenvalues before running the code?
        *   Now try a shear matrix like `[[1, 1], [0, 1]]`.
        *   For each matrix, `A`, and its resulting eigenvectors `v` and eigenvalues `λ`, verify the core equation by checking if `A @ v` is close to `λ * v`.
    *   **(10 mins):** Break.

*   **5:00 - 6:30 (90 mins): Deeper Exploration**
    *   **(80 mins):**
        *   Take the 90-degree rotation matrix `[[0, -1], [1, 0]]`. Use `np.linalg.eig()` on it. What do you notice about the results? (Hint: they're complex). Why does this make sense geometrically?
        *   Create a projection matrix that projects vectors onto the x-axis: `[[1, 0], [0, 0]]`. What are its eigenvalues? Does this make sense? (Hint: Vectors already on the x-axis are scaled by 1. Vectors on the y-axis are squashed to zero, i.e., scaled by 0).
    *   **(10 mins):** Break.

*   **6:30 - 7:00 (30 mins): Commit & Solidify**
    *   **(30 mins):**
        *   Add comments to your Python file explaining your experiments with eigenvectors.
        *   Commit your work to GitHub with a message like `feat: Explore eigenvalues and eigenvectors with NumPy`.

---

**Evening Session: Community & Review (1.5 hours)**

*   **7:00 - 7:45 (45 mins): Community & Wrap-up**
    *   **(30 mins):** Search for and read an article on "Eigenvectors and Eigenvalues in Principal Component Analysis (PCA)". This is a cornerstone of machine learning and one of the most direct applications of what you learned today.
    *   **(15 mins):** Create Anki flashcards for: `Eigenvector`, `Eigenvalue`, and the `Characteristic Equation`.

*   **7:45 - 8:00 (15 mins): Final Review & Planning**
    *   Quickly review today's notes and code.
    *   Look at tomorrow's goal from the roadmap. You're getting close to the end of the pure linear algebra section!
    *   Close your computer. Well done.
