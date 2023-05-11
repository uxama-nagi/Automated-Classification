import os
import csv
import re
import time
import argparse
from github import Github
from IPython.display import clear_output


def search_github(query, access_token, keywords, output_dir, num_row_per_file):
    # Access token to access the GitHub API
    g = Github(access_token, timeout=60)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the first CSV file for writing
    file_count = 1
    max_rows_per_file = num_row_per_file
    csvfile = open(os.path.join(output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
    fieldnames = ['timestamp', 'repository', 'username', 'message', 'type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Initialize row counter
    row_count = 0

    # Search for repositories matching the query
    results = g.search_repositories(query)
    counter = 1
    print(f"Total repositories found: {results.totalCount}")

    for repo in results:
        print(f"Processing Repo no : {counter} ---- {repo.full_name}")

        # Search for open issues
        try:
            issues = repo.get_issues(state='open')
            if issues is not None:
                for issue in issues:
                    if issue is not None:
                        if issue.title and issue.body is not None:
                            if any(re.search(r'\b{}\w*'.format(keyword), f'{issue.title.lower()} {issue.body.lower()}') for keyword in keywords):
                                writer.writerow({
                                    'timestamp': issue.created_at,
                                    'repository': repo.full_name,
                                    'username': issue.user.login,
                                    'message': issue.body,
                                    'type': "open_issue"
                                })
                                row_count += 1
                                if row_count >= max_rows_per_file:
                                    csvfile.close()
                                    file_count += 1
                                    csvfile = open(os.path.join(output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    # Reset row counter
                                    row_count = 0

        except Exception as e:
            time.sleep(90)
            continue

        # Search for closed issues
        try:
            issues = repo.get_issues(state='closed')
            if issues is not None:
                for issue in issues:
                    if issue is not None:
                        if issue.title and issue.body is not None:
                            if any(re.search(r'\b{}\w*'.format(keyword), f'{issue.title.lower()} {issue.body.lower()}') for keyword in keywords):
                                writer.writerow({
                                    'timestamp': issue.created_at,
                                    'repository': repo.full_name,
                                    'username': issue.user.login,
                                    'message': issue.body,
                                    'type': "closed_issue"
                                })
                                row_count += 1
                                if row_count >= max_rows_per_file:
                                    csvfile.close()
                                    file_count += 1
                                    csvfile = open(os.path.join(output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    # Reset row counter
                                    row_count = 0
                
        except Exception as e:
            time.sleep(90)
            continue

        # Search for open pull requests
        try:
            pulls = repo.get_pulls(state="open")
            if pulls is not None:
                for pull in pulls:
                    if pull is not None:
                        if pull.title and pull.body  is not None:
                            if any(re.search(r'\b{}\w*'.format(keyword), f'{pull.title.lower()} {pull.body.lower()}') for keyword in keywords):
                                writer.writerow({
                                    'timestamp': pull.created_at,
                                    'repository': repo.full_name,
                                    'username': pull.user.login,
                                    'message': pull.body,
                                    'type': "open_pull_request"
                                })
                                row_count += 1
                                if row_count >= max_rows_per_file:
                                    csvfile.close()
                                    file_count += 1
                                    csvfile = open(os.path.join(output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    # Reset row counter
                                    row_count = 0
        except Exception as e:
            time.sleep(90)
            continue
            
        # Search for closed pull requests
        try:
            pulls = repo.get_pulls(state="closed")
            if pulls is not None:
                for pull in pulls:
                    if pull is not None:
                        if pull.title and pull.body  is not None:
                            if any(re.search(r'\b{}\w*'.format(keyword), f'{pull.title.lower()} {pull.body.lower()}') for keyword in keywords):
                                writer.writerow({
                                    'timestamp': pull.created_at,
                                    'repository': repo.full_name,
                                    'username': pull.user.login,
                                    'message': pull.body,
                                    'type': "closed_pull_request"
                                })
                                row_count += 1
                                if row_count >= max_rows_per_file:
                                    csvfile.close()
                                    file_count += 1
                                    csvfile = open(os.path.join(output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    # Reset row counter
                                    row_count = 0
        except Exception as e:
            time.sleep(90)
            continue



        # Search for commits
        try:
            commits = repo.get_commits()
            if commits is not None:
                for commit in commits:
                    if commit is not None:
                        if commit.author and commit.commit.message is not None:
                            if any(re.search(r'\b{}\w*'.format(keyword), f'{commit.commit.message.lower()}') for keyword in keywords):
                                writer.writerow({
                                    'timestamp': commit.commit.author.date,
                                    'repository': repo.full_name,
                                    'username': commit.commit.author.name,
                                    'message': commit.commit.message,
                                    'type': "commit"
                                })
                                row_count += 1
                                if row_count >= max_rows_per_file:
                                    csvfile.close()
                                    file_count += 1
                                    csvfile = open(os.path.join(output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    # Reset row counter
                                    row_count = 0
        except Exception as e:
            time.sleep(90)
            continue

        counter += 1
        clear_output()

    total = (file_count * num_row_per_file) + row_count
    print(f"{total} data from {counter} repositories is mined and stored successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for data in GitHub repositories")
    parser.add_argument("query", type=str, help="The query to search for repositories on GitHub e.g. language:java stars:>100")
    parser.add_argument("access_token", type=str, help="GitHub API Access Token")
    parser.add_argument("keywords", type=str, nargs="+", help="a list of keywords to search")
    parser.add_argument("output_dir", type=str, help="The directory where the CSV files will be saved")
    parser.add_argument("num_row_per_file", type=int, help="Maximum Number of Rows per file")
    args = parser.parse_args()

    search_github(args.query, args.access_token, args.keywords, args.output_dir, args.num_row_per_file)
