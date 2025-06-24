# Plan for Day 2: The Power of Matrices - Determinants & Inverses

**Morning Session: Theory & Intuition (2.5 hours)**

*   **0:00 - 1:00 (60 mins): Core Matrix Concepts**
    *   **(50 mins):** Watch the next videos in 3Blue1Brown's "Essence of Linear Algebra" series.
        *   *The determinant*
        *   *Inverse matrices, column space and null space*
        *   *Nonsquare matrices as transformations between dimensions*
    *   **(10 mins):** Break. Think about the determinant. What does the *sign* of the determinant mean? What does a determinant of zero signify?

*   **1:00 - 2:00 (60 mins): Active Recall & Deeper Understanding**
    *   **(50 mins):** In your notebook, write down answers from memory:
        *   What is the geometric meaning of a matrix's determinant?
        *   What does it mean for a matrix to be "invertible"? What is the relationship between the inverse and the determinant?
        *   What's the difference between a "column space" and a "null space"? How do they relate to the solutions of a system of equations `Ax = v`?
    *   Re-watch sections as needed to solidify your understanding.
    *   **(10 mins):** Break.

*   **2:00 - 2:30 (30 mins): Formalize with Reading**
    *   **(30 mins):** Read the corresponding sections in *Linear Algebra Done Right* covering determinants and invertibility. Connect the formal definitions to the 3Blue1Brown intuitions.

---

**Mid-day Break (1 hour)**

---

**Afternoon Session: Practice & Implementation (4 hours)**

*   **3:30 - 5:00 (90 mins): Hands-on with NumPy**
    *   **(80 mins):** Open your `q1_foundations/linear_algebra.py` file.
        *   **Determinant:** Implement a function to calculate the determinant of a 2x2 and a 3x3 matrix from scratch. The formula is straightforward for these.
        *   **Inverse:** Implement a function to calculate the inverse of a 2x2 matrix from scratch.
        *   Use NumPy's `np.linalg.det()` and `np.linalg.inv()` to check your work on several test matrices. Note how much easier it is with a library!
    *   **(10 mins):** Break.

*   **5:00 - 6:30 (90 mins): Deeper Exploration**
    *   **(80 mins):**
        *   Create a singular (non-invertible) matrix (e.g., where one column is a multiple of another).
        *   Try to calculate its determinant and inverse using both your functions and NumPy's. What happens? Why?
        *   Explore the concepts of column space and null space using NumPy. Can you find a vector `v` that is in the column space of a matrix `A`? (Hint: `Ax=v` has a solution). Can you find a vector `x` (other than zero) that is in the null space of `A`? (Hint: `Ax=0`).
    *   **(10 mins):** Break.

*   **6:30 - 7:00 (30 mins): Commit & Solidify**
    *   **(30 mins):**
        *   Add comments and docstrings to your new functions.
        *   Commit your work to GitHub with a clear message like `feat: Implement determinant and inverse functions for 2x2 matrices`.

---

**Evening Session: Community & Review (1.5 hours)**

*   **7:00 - 7:45 (45 mins): Community & Wrap-up**
    *   **(30 mins):** Search for and read a blog post or article on "Why is the matrix determinant useful in machine learning?". See how this abstract concept connects to future topics.
    *   **(15 mins):** Create Anki flashcards for: `Determinant`, `Inverse Matrix`, `Singular Matrix`, `Column Space`, `Null Space`.

*   **7:45 - 8:00 (15 mins): Final Review & Planning**
    *   Quickly review today's notes and code.
    *   Look at tomorrow's goal from the roadmap: Eigenvectors and Eigenvalues. Get excited!
    *   Close your computer. Well done. 