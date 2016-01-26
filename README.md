# cs181-s16-homeworks
Seed repository for homeworks in cs181, s16

Welcome to CS181! This repository will house your homework assignments.

If you are reading this, the name of the repository should be cs181-s16-homeworks-YOUR GITHUB ID. If that's what you see, you should already be logged into your github account and have your own private repository (this one) for your homeworks. You're good to read on.

If you don't see your github id at the end of the repository name, you need to find the link on [canvas](https://canvas.harvard.edu/courses/9660/) to create your own private repository.

If you don't know what any of this means, stop drop and contact a TF. These instructions only apply to your private repo.

## How this repository works

This repository is a copy of a "seed" repository maintained by the TFs. Right now there should be five empty folders in this repository. Throughout this semester, your TFs will add homework assignments to the seed repository, and you will be responsible for copying them into your repository ("fetch" and "merge" in git-speak). Assuming you've already cloned your repository locally, you get the new files by:
```
git remote add seed_repo https://github.com/harvard-ml-courses/cs181-s16-homeworks.git # only needs to be done once
git fetch seed_repo
git merge seed_repo/master -m "Fetched new assignment"
```
These commands (1) tell your local git repository where the seed repo is (and calls is "seed_repo"), (2) gets that repo from github.com, and (3) merges it with your local files. A final, fourth, step would be to push to your remote repository so it shows up on the web and your TFs can see it.

In fact, try that now to make sure you don't get any errors, and contact a TF via canvas if you do.

## How homeworks work

For each assignment, you'll be given a number of files. One of them will be a .tex file, with some blanks for where the solutions go. That's where the solutions go. When you complete the homeworks, you need to push the .tex file to this repository and submit the pdfs via canvas.

## Caveats

This is a new system that we're trying out, so if are having any trouble, please reach out to a TF. If you need help acclimatizing to git, again, reach out. We're here to help.

Enjoy!
