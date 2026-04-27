# GitHub Upload Notes

This machine can reach `api.github.com:443`, but direct `git push` to `github.com:443` may fail or time out.

For this project, the successful upload path was:

1. Create or verify the target GitHub repository.
2. If the repository is empty, initialize it with a small placeholder file through the GitHub connector or GitHub API.
3. Initialize the local project as a git repository and commit the intended files.
4. Use Git Credential Manager to retrieve the existing GitHub HTTPS credential without printing the token.
5. Use GitHub REST API endpoints to:
   - read `refs/heads/main`
   - create a tree from `git ls-files`
   - create a commit using that tree
   - update `refs/heads/main`
6. Verify with the GitHub connector by fetching an uploaded file and the commit.

Do not upload local caches, IDE files, training outputs, checkpoints, or generated media. The project `.gitignore` excludes those paths.

The repository used for this upload was:

```text
https://github.com/xxxllli/qwen3_vl_fitness_sft.git
```

The first full upload commit was:

```text
c6462670f2d5abf9f0d0008f668b4b0f3e186f9e
```

Use the API upload route first next time if `Test-NetConnection github.com -Port 443` fails but `Test-NetConnection api.github.com -Port 443` succeeds.
