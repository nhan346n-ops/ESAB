pipeline {
	agent any
	stages {
        stage('Pre-build') {
            steps {
                script {
                    currentBuild.displayName = "#${env.BUILD_NUMBER} ${env.GIT_BRANCH}"
                }
                script {
                    echo "Retrieve email list"
                    withCredentials([
                        string(credentialsId: 'NOTIFICATION_GLOBE_MAIL_LIST', variable: 'mail_list')]) {
                        env.NOTIFICATION_MAIL_LIST = mail_list
                    }
                }
            }
        }
		stage('Create environment') {
			steps {
				echo 'Building anaconda environment'
				echo 'Path = ${Path}'
                bat label: 'create conda environments', script: '''call C:\\Tools\\Miniconda3\\Library\\bin\\conda.bat activate
							cd requirements
							python create_anaconda_environments.py -target test
							cd ..
							'''
			}
		}
		stage('test'){
			steps{
				bat label: 'cleaning python caches', script: '''call C:\\Tools\\Miniconda3\\Library\\bin\\conda.bat activate pyat_test
                            git reset --hard
                            git clean -fx
                            '''
                bat label: 'Install pyat in local editable mode', script: '''call C:\\Tools\\Miniconda3\\Library\\bin\\conda.bat activate pyat_test
                            python -m pip install -e . --no-cache-dir
                            '''
				bat label: 'Running pylint on core package', script: '''call C:\\Tools\\Miniconda3\\Library\\bin\\conda.bat activate pyat_test
							pylint pyat
							'''
				bat label: 'Running pytest', script: '''call C:\\Tools\\Miniconda3\\Library\\bin\\conda.bat activate pyat_test
							python.exe -m pytest --verbose --junit-xml test-reports/results.xml -o junit_family=xunit2 --maxfail=150 --ignore=heightmap_interpolation/tests/test_div_of_grad.py
							'''
			}
		}
    }
	post {
        always {
            // Archive unit tests for the future
            junit allowEmptyResults: true, testResults: 'pyat/test-reports/results.xml'
        }
        failure {
            emailext (
            subject: "Jenkins FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}] ${env.GIT_BRANCH}'",
            attachLog: true,
            body: """BUILD FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}] ${env.GIT_BRANCH}': Check console output at ${env.BUILD_URL}
                (see Log attached for details)
            """,
            to: "${env.NOTIFICATION_MAIL_LIST}"
            )
        }
        success {
            emailext (
                subject: "Jenkins Pyat SUCCESS: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}] ${env.GIT_BRANCH}'",
                attachLog: true,
                body: """BUILD SUCCESS: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}] ${env.GIT_BRANCH}'
                    Check console output at ${env.BUILD_URL}
                """,
                to: "${env.NOTIFICATION_MAIL_LIST}"
                )
        }
    }
}
