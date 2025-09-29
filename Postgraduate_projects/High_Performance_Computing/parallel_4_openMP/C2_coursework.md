# Assessed Coursework - Are you smarter than Copilot?

## Guidance
The coursework will be submitted via a GitLab repository which we will create for you.  You should place all your code and your report in this repository. The report should be in PDF format in a folder called "report".  You will be provided access to the repository until 11:59pm on Friday the 4th of April.  After this you will lose access which will constitute submission of your work.

## The problem
I asked Copilot which simple 2D physics simulations it could create.  It replied that it could simulate heat diffusion (amongst many others, mostly from fluid dynamics, which when tested it definitely couldn't do) and so I asked it to create a project that simulated heat diffusion, with the files needed to build it.  

In the folder you will find the code it generated.  

Your coursework is to optimise and parallelise this code.  You do not need to improve the underlying algorithm, or its stability, only the implementation of it. You only need to make sure that your version returns the same results as the baseline code.

The code should be simple to use, written in C++, and come with appropriate example submission scripts for CSD3.  The code should be able effectively use the resources available on two intel Icelake nodes.

## ** Warning **
Achieving maximum performance for code can be a long and arduous process.  You should aim for a small (order of 2x) performance gain over the baseline code on a fixed number of cores, and some working implementation of parallelisation.  It will be easy to go down a rabbit hole on this project, so you must be careful manage your time to ensure you are able to write a report and submit your work by the deadline (and complete your other courseworks).  

I am not looking for perfection and you will be able to achieve high marks for projects that fall short of the maximum possible performance.

## Criteria
The goal of this coursework project is for you to demonstrate that you understand parallelisation and optimisation practices for High Performance Computing.  

The key skills you will be expected to demonstrate are:
- Optimisation of the routines for simulating heat diffusion
- Flexible and robust MPI domain decomposition of the grid and communication of the boundaries at each step.
- Effective Thread parallelisation using OpenMP
- Rigorous performance testing with appropriate plots and tables demonstrating the improvement of each optimisation
- Detailed analysis of the scaling of the code with the number of MPI ranks and threads used in parallelisation.

While the focus of this project is performance we still expect you to demonstrate the following software development best practice:
- Writing clear readable code that is compliant with a common style guide and uses suitable build management tools. 
- Providing appropriate documentation that is compatible with auto-documentation tools like Doxygen.
- The project must be well structured (sensible folder structure, readme, licence etc..) following standard best practice.
- Appropriate and robust unit tests for automatic validation of your code.
- Uses appropriate version control best practice, including branching for development and testing.  Given the parallel nature of the project, containerisation is not required.

As this project is designed to test your optimisation skills, **no linear algebra or image processing libraries should be used**.  The expectation is that you will only need: standard C++libraries, MPI, and OpenMP

## Submission
This repository will contain the working code for the project. This should be accompanied by a short report describing the project and its development. You should ensure your report is logically structured and touches on the points mentioned above in the assessment criteria. Specifically, I would expect reports to cover the following topics, with the two middle sections being the largest:

- Short Introduction
- Development, Experimentation, and Profiling of Optimisation steps with discussion and explanation of the observed results
- Development, Experimentation, and Profiling of Parallelisation with discussion and explanation of the observed results
- Summary

The report should be written in latex, with the generated PDF of the report placed in a folder called "report" in the repository.

**Your report should not exceed 3000 words (including tables, figure captions and appendices but excluding references); please indicate the word count on the front cover. You are reminded to comply with the requirements given in the Course Handbook regarding the use of, and declaration of use of, auto-generation tools.**
