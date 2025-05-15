from api.routers.auth import pwd_context
print(pwd_context.hash('test'))