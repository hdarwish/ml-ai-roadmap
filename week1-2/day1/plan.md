# Plan for Day 1: Linear Algebra Foundations

**Morning Session: Theory & Intuition (2.5 hours)**

*   **0:00 - 1:00 (60 mins): Foundational Concepts**
    *   **(50 mins):** Watch the first three videos of 3Blue1Brown's "Essence of Linear Algebra" series.
        *   *Vectors, what even are they?*
        *   *Linear combinations, span, and basis vectors*
        *   *Linear transformations and matrices*
    *   **(10 mins):** Break. Walk around and let the concepts sink in.

*   **1:00 - 2:00 (60 mins): Active Recall & Deeper Understanding**
    *   **(50 mins):** In a notebook (physical or digital), write down answers to these questions *from memory*:
        *   How would you explain a vector to a 5th grader? What about its geometric vs. data representation?
        *   What is "span"? What's the difference between a "basis vector" and any other vector?
        *   What is a matrix, really? What does it *do* to a vector?
    *   Now, re-watch any sections from the videos that were unclear and refine your answers.
    *   **(10 mins):** Break.

*   **2:00 - 2:30 (30 mins): Formalize with Reading**
    *   **(30 mins):** Read the first few sections of Chapter 1 in *Linear Algebra Done Right* (or your chosen book). Your goal is to connect the formal mathematical language (e.g., vector spaces, subspaces) to the visual intuition you just built.

---

**Mid-day Break (1 hour)**

---

**Afternoon Session: Practice & Implementation (4 hours)**

*   **3:30 - 4:30 (60 mins): Environment Setup**
    *   **(50 mins):**
        1.  Go to GitHub and create your new public portfolio repository as the roadmap suggests.
        2.  Set up your local Python environment (e.g., using `venv`).
        3.  Install NumPy: `pip install numpy`.
        4.  In your new repository, create a folder named `q1_foundations` and a file inside called `linear_algebra.py`. This is where your from-scratch functions will live.
    *   **(10 mins):** Break.

*   **4:30 - 6:30 (120 mins): Hands-on with NumPy**
    *   **(50 mins):** Open your `linear_algebra.py` file. Follow a NumPy tutorial or its documentation to get comfortable with the basics.
        *   Create vectors (1D arrays) and matrices (2D arrays).
        *   Check their `.shape` and `.dtype`.
        *   Practice vector addition, subtraction, and scalar multiplication.
    *   **(10 mins):** Break.
    *   **(50 mins):** Start implementing the core operations from the roadmap.
        *   **Dot Product:** Write a function `vector_dot_product(v1, v2)` that takes two Python lists and calculates their dot product. Then, write a separate function that does the same thing but uses NumPy's `@` operator or `np.dot()`.
        *   **Transpose:** Write a function that takes a matrix (list of lists) and returns its transpose. Then, find the NumPy equivalent (`.T`).
    *   **(10 mins):** Break.

*   **6:30 - 7:30 (60 mins): Commit & Solidify**
    *   **(60 mins):**
        *   Add comments to your code explaining what each function does.
        *   Use `git` to commit your work to your GitHub repository. Write a clear commit message like `feat: Implement vector dot product and matrix transpose`. This starts another key habit.
        *   Experiment with matrix-vector multiplication in NumPy. See if you can predict the shape of the output before you run the code.

---

**Evening Session: Community & Review (1.5 hours)**

*   **7:30 - 8:15 (45 mins): Community & Wrap-up**
    *   **(30 mins):** As the roadmap suggests, spend time with the community. Go to a site like Reddit (`r/learnmachinelearning`) or search for "Linear Algebra for Data Science" blogs. Read about someone else's journey or explanation. This exposes you to different perspectives.
    *   **(15 mins):** Create your first set of **Anki** flashcards. Good cards for today would be: `Vector`, `Matrix`, `Span`, `Basis`, `Dot Product`, `Transpose`. On one side, the term; on the other, a simple definition *in your own words*.

*   **8:15 - 8:30 (15 mins): Final Review & Planning**
    *   Quickly review your notes and the code you wrote.
    *   Look at tomorrow's goal: continuing with matrix operations like determinants and inverses.
    *   Close your computer. You're done for the day.